//! The trainer: initialization, the sampler of `ArithmeticIterator`, a step on a grid of
//! threads that each take a share of the batch, and exhaustive evaluation.

use crate::adam::{self, Hyper};
use crate::arith;
use crate::data::{ANSWER_POS, EOS_POS, Equation};
use crate::layout::Dims;
use crate::model::{Model, Transposed, Work, decide, softmax};
use crate::mt::Mt;

/// The seed of the stream that draws the initial parameters of run `seed`.
pub fn init_seed(seed: u32) -> u32 {
    1_000_003u32.wrapping_mul(seed.wrapping_add(1))
}

/// The seed of the stream that draws the batches of run `seed`.
pub fn sampler_seed(seed: u32) -> u32 {
    2_000_003u32.wrapping_mul(seed.wrapping_add(1))
}

/// PyTorch's default initialization of the source's modules, drawn in the order of the flat
/// parameter vector from the generator `rng`: the embedding standard normal; each linear map
/// uniform on `[-1/sqrt(fan_in), 1/sqrt(fan_in)]`, which is `kaiming_uniform_(a = sqrt(5))`;
/// gains 1 and biases 0.
pub fn initialize(dims: Dims, rng: &mut Mt) -> Vec<f32> {
    let (d, dff) = (dims.d, dims.dff);
    let mut p = vec![0f32; dims.len()];
    let uniform = |slice: &mut [f32], fan_in: usize, rng: &mut Mt| {
        let bound = 1.0 / (fan_in as f64).sqrt();
        for value in slice {
            *value = ((2.0 * rng.double() - 1.0) * bound) as f32;
        }
    };
    for value in &mut p[..dims.vocab * d] {
        *value = rng.normal() as f32;
    }
    for l in 0..dims.layers {
        let at = dims.block(l);
        uniform(&mut p[at.wq..at.wq + d * d], d, rng);
        uniform(&mut p[at.wkv..at.wkv + 2 * d * d], d, rng);
        uniform(&mut p[at.wo..at.wo + d * d], d, rng);
        p[at.g1..at.g1 + d].fill(1.0);
        uniform(&mut p[at.w1..at.w1 + d * dff], d, rng);
        uniform(&mut p[at.w2..at.w2 + dff * d], dff, rng);
        p[at.g2..at.g2 + d].fill(1.0);
    }
    uniform(&mut p[dims.wout()..], d, rng);
    p
}

/// The batch size of `calculate_batchsize` at its default hint: `min(512, ceil(n / 2))`.
pub fn batch_size(n: usize) -> usize {
    512.min(n.div_ceil(2))
}

/// The batches of `ArithmeticIterator` over a training set of `n` equations: a fresh uniform
/// permutation for each epoch, cut into batches in order, the last one short.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Sampler {
    /// The generator of the permutations.
    pub rng: Mt,
    /// The epoch's permutation.
    pub perm: Vec<u32>,
    /// Batches taken from it.
    pub index: usize,
    /// The batch size.
    pub batch: usize,
}

impl Sampler {
    /// The sampler of the stream seeded with `seed`, its first permutation drawn.
    pub fn new(seed: u32, n: usize, batch: usize) -> Sampler {
        let mut sampler = Sampler { rng: Mt::new(seed), perm: (0..n as u32).collect(), index: 0, batch };
        sampler.rng.shuffle(&mut sampler.perm);
        sampler
    }

    /// The next batch, as indices of the training set.
    pub fn next(&mut self) -> Vec<u32> {
        let n = self.perm.len();
        if self.index * self.batch > n - 1 {
            self.perm = (0..n as u32).collect();
            self.rng.shuffle(&mut self.perm);
            self.index = 0;
        }
        let begin = self.index * self.batch;
        self.index += 1;
        self.perm[begin..n.min(begin + self.batch)].to_vec()
    }
}

/// What an exhaustive evaluation of a set of equations finds.
#[derive(Clone, Debug, Default)]
pub struct Eval {
    /// The mean cross-entropy of the answer.
    pub ce_answer: f64,
    /// The mean cross-entropy of the closing `<|eos|>`.
    pub ce_eos: f64,
    /// For each equation, the token predicted for the answer: the first of the largest logits
    /// over the whole vocabulary.
    pub predictions: Vec<u8>,
    /// For each equation, the answer's logit less the largest logit of another token.
    pub margins: Vec<f32>,
    /// For each equation, whether the closing `<|eos|>` is predicted.
    pub eos_right: Vec<bool>,
    /// For each equation, whether the answer is predicted.
    pub right: Vec<bool>,
}

impl Eval {
    /// The fraction of equations whose answer is predicted.
    pub fn accuracy(&self) -> f64 {
        self.right.iter().filter(|&&r| r).count() as f64 / self.right.len() as f64
    }

    /// The fraction with both the answer and the closing token predicted: the source's accuracy.
    pub fn accuracy_source(&self) -> f64 {
        let both = self.right.iter().zip(&self.eos_right).filter(|&(&a, &b)| a && b).count();
        both as f64 / self.right.len() as f64
    }

    /// The source's loss: the mean over both supervised tokens.
    pub fn loss_source(&self) -> f64 {
        (self.ce_answer + self.ce_eos) / 2.0
    }

    /// The margin below which the fraction `q` of the equations lie, by rank.
    pub fn margin_quantile(&self, q: f64) -> f32 {
        let mut sorted = self.margins.clone();
        sorted.sort_by(f32::total_cmp);
        sorted[((sorted.len() - 1) as f64 * q).round() as usize]
    }
}

/// The tokens read and the targets at the positions `sup`, example by example.
fn encode(part: &[Equation], sup: &[usize], ctx: usize) -> (Vec<usize>, Vec<usize>) {
    let tokens = part.iter().flat_map(|eq| eq.tokens()[..ctx].to_vec()).collect();
    let targets = part.iter().flat_map(|eq| sup.iter().map(|&pos| eq.target(pos)).collect::<Vec<_>>()).collect();
    (tokens, targets)
}

/// Examples in a chunk of a batch. A batch is cut into chunks of this size in order, each
/// chunk's gradient is a sum over its rows in order, and the batch's is the sum of the chunks'
/// in order: the floating-point result is the same on any number of threads.
pub const CHUNK: usize = 32;

/// A model in training.
pub struct Trainer {
    /// The sizes.
    pub dims: Dims,
    /// The supervised positions.
    pub sup: Vec<usize>,
    /// The optimizer's constants.
    pub hyper: Hyper,
    /// The parameters.
    pub p: Vec<f32>,
    /// The first moments.
    pub m: Vec<f32>,
    /// The second moments.
    pub v: Vec<f32>,
    /// Updates made.
    pub step: u64,
    /// The gradient of the last batch.
    pub grad: Vec<f32>,
    pe: Vec<f32>,
    wt: Transposed,
    works: Vec<Work>,
    sums: Vec<f64>,
    threads: usize,
    flush: bool,
}

impl Trainer {
    /// A trainer of the parameters `p`, with zero moments, on `threads` threads, which share
    /// the chunks of a batch: the result does not depend on how many they are.
    pub fn new(dims: Dims, sup: Vec<usize>, hyper: Hyper, p: Vec<f32>, threads: usize) -> Trainer {
        assert!(crate::gemm::supported(), "the kernel needs AVX2 and FMA");
        assert_eq!(p.len(), dims.len());
        assert!(threads > 0, "a step needs a thread");
        let zeros = vec![0f32; dims.len()];
        Trainer {
            dims,
            sup,
            hyper,
            p,
            m: zeros.clone(),
            v: zeros.clone(),
            step: 0,
            grad: zeros,
            pe: dims.positions(),
            wt: Transposed::new(dims),
            works: Vec::new(),
            sums: Vec::new(),
            threads,
            flush: false,
        }
    }

    /// Sets whether the calling thread and every thread of a step or of an evaluation flush
    /// subnormal numbers to zero (`PROTOCOL.md`, Amendment 2). A trainer starts with gradual
    /// underflow, the source's arithmetic.
    pub fn set_flush(&mut self, on: bool) {
        self.flush = on;
        arith::flush_subnormals(on);
    }

    /// The forward and backward passes of the batch, chunk by chunk: each chunk's gradient is
    /// left in its `Work`, and its sum of cross-entropies in `self.sums`. The loss of the
    /// batch, its mean cross-entropy over the equations and the supervised positions.
    fn chunks(&mut self, batch: &[Equation]) -> f64 {
        self.wt.update(self.dims, &self.p);
        let model = Model { dims: self.dims, p: &self.p, pe: &self.pe, wt: &self.wt, sup: &self.sup };
        let count = batch.len() * self.sup.len();
        let (scale, ctx) = (1.0 / count as f32, self.dims.ctx);
        let parts = batch.len().div_ceil(CHUNK);
        if self.works.len() < parts {
            self.works.resize_with(parts, Work::default);
        }
        self.sums.clear();
        self.sums.resize(parts, 0.0);
        let mut groups: Vec<Vec<(&[Equation], &mut Work, &mut f64)>> = (0..self.threads).map(|_| Vec::new()).collect();
        for (i, ((part, work), sum)) in
            batch.chunks(CHUNK).zip(self.works.iter_mut()).zip(self.sums.iter_mut()).enumerate()
        {
            groups[i % self.threads].push((part, work, sum));
        }
        let flush = self.flush;
        std::thread::scope(|scope| {
            for group in groups {
                let model = &model;
                scope.spawn(move || {
                    arith::flush_subnormals(flush);
                    for (part, work, sum) in group {
                        let (tokens, targets) = encode(part, model.sup, ctx);
                        model.forward(work, &tokens);
                        *sum = model.backward(work, &tokens, &targets, scale).iter().sum();
                    }
                });
            }
        });
        self.sums.iter().sum::<f64>() / count as f64
    }

    /// The loss of the batch with its gradient into `self.grad`: the chunks' gradients summed
    /// in order.
    pub fn gradient(&mut self, batch: &[Equation]) -> f64 {
        let loss = self.chunks(batch);
        let parts: Vec<&[f32]> = self.works[..self.sums.len()].iter().map(|w| &w.grad[..]).collect();
        reduce(&mut self.grad, &parts, 0);
        loss
    }

    /// One update on the batch: its loss before the update. The sum of the chunks' gradients
    /// and the optimizer's update act on each parameter alone, so the threads share the
    /// parameters by ranges.
    pub fn update(&mut self, batch: &[Equation]) -> f64 {
        let loss = self.chunks(batch);
        self.step += 1;
        let parts: Vec<&[f32]> = self.works[..self.sums.len()].iter().map(|w| &w.grad[..]).collect();
        let size = self.p.len().div_ceil(self.threads);
        let (hyper, step, flush) = (self.hyper, self.step, self.flush);
        std::thread::scope(|scope| {
            let ranges = self.p.chunks_mut(size).zip(self.m.chunks_mut(size)).zip(self.v.chunks_mut(size));
            for (i, (((p, m), v), g)) in ranges.zip(self.grad.chunks_mut(size)).enumerate() {
                let parts = &parts;
                scope.spawn(move || {
                    arith::flush_subnormals(flush);
                    reduce(g, parts, i * size);
                    adam::update(&hyper, step, p, m, v, g);
                });
            }
        });
        loss
    }

    /// The answer and the closing token of every equation of `set`, by the current parameters.
    pub fn evaluate(&mut self, set: &[Equation]) -> Eval {
        let sup = [ANSWER_POS, EOS_POS];
        let model = Model { dims: self.dims, p: &self.p, pe: &self.pe, wt: &self.wt, sup: &sup };
        let (ctx, vocab) = (self.dims.ctx, self.dims.vocab);
        if self.works.len() < self.threads {
            self.works.resize_with(self.threads, Work::default);
        }
        let (share, flush) = (set.len().div_ceil(self.threads).max(1), self.flush);
        let rows: Vec<Vec<(f64, f64, u8, f32, bool)>> = std::thread::scope(|scope| {
            let handles: Vec<_> = set
                .chunks(share)
                .zip(self.works.iter_mut())
                .map(|(mine, work)| {
                    let model = &model;
                    scope.spawn(move || {
                        arith::flush_subnormals(flush);
                        let mut rows = Vec::with_capacity(mine.len());
                        for part in mine.chunks(128) {
                            let (tokens, targets) = encode(part, model.sup, ctx);
                            model.forward(work, &tokens);
                            for b in 0..part.len() {
                                let answer = &work.logits[2 * b * vocab..][..vocab];
                                let eos = &work.logits[(2 * b + 1) * vocab..][..vocab];
                                let (predicted, margin) = decide(answer, targets[2 * b]);
                                let eos_right = decide(eos, targets[2 * b + 1]).0 == targets[2 * b + 1];
                                let ces = (softmax(answer, targets[2 * b]).0, softmax(eos, targets[2 * b + 1]).0);
                                rows.push((ces.0, ces.1, predicted as u8, margin, eos_right));
                            }
                        }
                        rows
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().expect("a thread of the evaluation panicked")).collect()
        });
        let mut eval = Eval::default();
        let (mut sum_answer, mut sum_eos) = (0f64, 0f64);
        for (ce_answer, ce_eos, predicted, margin, eos_right) in rows.into_iter().flatten() {
            sum_answer += ce_answer;
            sum_eos += ce_eos;
            let target = set[eval.predictions.len()].target(ANSWER_POS);
            eval.right.push(predicted as usize == target);
            eval.predictions.push(predicted);
            eval.margins.push(margin);
            eval.eos_right.push(eos_right);
        }
        eval.ce_answer = sum_answer / set.len() as f64;
        eval.ce_eos = sum_eos / set.len() as f64;
        eval
    }
}

/// `g[i]` becomes the sum over `parts`, in order, of their elements `at + i`.
fn reduce(g: &mut [f32], parts: &[&[f32]], at: usize) {
    g.fill(0.0);
    let n = g.len();
    for part in parts {
        for (total, value) in g.iter_mut().zip(&part[at..at + n]) {
            *total += *value;
        }
    }
}

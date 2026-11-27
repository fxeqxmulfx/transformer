//! Argand's complex decoder and tied vocabulary readout on Kernite.
//!
//! One model trains here: skip and write gates around every sublayer, score
//! multipliers starting at `1 + 0i`, and the clipped momentum steps of the
//! Lean `fullGradientMomentumRun` on the full training gradient plus
//! `PENALTY * x`. Earlier residual, score and optimizer variants are in
//! experiments/phasor_value_transport at commit 7399c3bb.
mod gemm;
mod kernels;
mod residual;
mod scores;
mod tape;
mod weights;

pub(crate) use crate::Result;
use crate::{base_ops, optimizer_kernels, random::Random, stable_loss, suite::Shape};
use kernite_cuda::{Buffer, Cuda, Grid, Ptr};
use residual::Sublayer;
use tape::{Context, Pair, Tape};
use weights::Parameters;

type F = Buffer<f32>;
type P = Ptr<f32>;

/// The checkpoint identity of the model.
pub const MODEL: &str =
    "argand-skip-write-gate-single-stream-tied-vocab-fp32-s-init-one-sgd-momentum-full-batch";

/// The coefficient λ of the penalty `(λ/2)‖x‖²` in the full-gradient objective.
pub const PENALTY: f32 = 1e-4;

/// The initial score multiplier `s = 1 + 0i` of every layer.
const SCORE_INIT: [f32; 2] = [1.0, 0.0];

fn grid(count: usize) -> Grid {
    Grid::new(count.div_ceil(256) as u32, 256)
}

pub struct Scoring {
    pub max_batch: usize,
    pub vocab: usize,
}

pub struct Engine {
    pub config: Scoring,
    pub predictions: Buffer<u32>,
    pub logits: F,
    pub cuda: Cuda,
    shape: Shape,
    repeat: usize,
    parameters: Parameters,
    tape: Tape,
    kernel: kernels::device::Kernels,
    gemm: gemm::device::Kernels,
    /// The partial products of a split `k`: at most four tiles of the
    /// tiled product for each multiprocessor.
    workspace: F,
    scores: scores::device::Kernels,
    embed: base_ops::kernels::Kernels,
    loss: stable_loss::kernels::Kernels,
    optimizer: optimizer_kernels::kernels::Kernels,
    inputs: Buffer<u32>,
    labels: Buffer<u32>,
    loss_values: F,
    logit_grad: F,
    rotations: F,
    last: Option<(Pair, Pair, usize, usize)>,
}

impl Engine {
    pub fn new(shape: Shape, repeat: usize, seed: u32) -> Result<Self> {
        if shape.width == 0
            || shape.heads == 0
            || !shape.width.is_multiple_of(shape.heads)
            || shape.layers == 0
            || repeat == 0
            || shape.layers.checked_mul(repeat).is_none()
        {
            return Err("invalid Argand model dimensions".into());
        }
        let cuda = Cuda::new()?;
        let rows = shape.max_batch * shape.max_length;
        let vocab = shape.vocab;
        let logits = rows * vocab;
        let config = Scoring {
            max_batch: shape.max_batch,
            vocab,
        };
        Ok(Self {
            config,
            repeat,
            parameters: Parameters::new(&cuda, &shape, seed)?,
            tape: Tape::default(),
            kernel: kernels::device::Kernels::load(&cuda)?,
            gemm: gemm::device::Kernels::load(&cuda)?,
            workspace: cuda.alloc(
                8 * cuda.multiprocessors() as usize * gemm::device::TILE * gemm::device::TILE,
            )?,
            scores: scores::device::Kernels::load(&cuda)?,
            embed: base_ops::kernels::Kernels::load(&cuda)?,
            loss: stable_loss::kernels::Kernels::load(&cuda)?,
            optimizer: optimizer_kernels::kernels::Kernels::load(&cuda)?,
            inputs: cuda.alloc(rows)?,
            labels: cuda.alloc(rows)?,
            predictions: cuda.alloc(rows)?,
            logits: cuda.alloc(logits)?,
            logit_grad: cuda.alloc(logits)?,
            loss_values: cuda.alloc(rows)?,
            rotations: cuda.alloc(3 * shape.max_length * (shape.width / shape.heads))?,
            last: None,
            shape,
            cuda,
        })
    }

    /// The number of supervised positions: labels below the vocabulary size.
    pub fn labeled(&self, labels: &[u32]) -> usize {
        labels
            .iter()
            .filter(|&&y| (y as usize) < self.config.vocab)
            .count()
    }

    pub fn forward(
        &mut self,
        tokens: &[u32],
        labels: &[u32],
        batch: usize,
        length: usize,
    ) -> Result<()> {
        self.forward_normalized(tokens, labels, batch, length, None)
    }

    /// The loss gradient divides by `normalizer`, by default this batch's
    /// own label count.
    fn forward_normalized(
        &mut self,
        tokens: &[u32],
        labels: &[u32],
        batch: usize,
        length: usize,
        normalizer: Option<usize>,
    ) -> Result<()> {
        let rows = batch * length;
        if batch == 0
            || batch > self.shape.max_batch
            || length == 0
            || length > self.shape.max_length
            || tokens.len() != rows
            || labels.len() != rows
            || tokens.iter().any(|&id| id as usize >= self.config.vocab)
        {
            return Err("invalid Argand input batch".into());
        }
        let own = self.labeled(labels).max(1);
        let normalizer = match normalizer {
            None => own,
            Some(total) if total >= own => total,
            Some(_) => return Err("the full-batch label count is below a microbatch's".into()),
        };
        self.last = None;
        self.inputs
            .part_mut(..rows)
            .ok_or("input capacity")?
            .write(tokens)?;
        self.labels
            .part_mut(..rows)
            .ok_or("label capacity")?
            .write(labels)?;
        self.tape.reset();
        let width = self.shape.width;
        let input = self.tape.allocate(&self.cuda, rows * 2 * width)?;
        // SAFETY: all input IDs and lengths were validated above.
        unsafe {
            self.embed.embed(
                grid(input.len),
                input.x,
                self.parameters.embedding().x,
                self.inputs.ptr(),
                (2 * width) as u32,
                input.len as u32,
            )?;
        }
        let hidden = self.inner(input, batch, length)?;
        let scores = self.readout(hidden, batch, length)?;
        // SAFETY: stable loss receives one complete vocabulary row per position.
        unsafe {
            self.loss.loss(
                Grid::new(rows.div_ceil(8) as u32, 256),
                self.logit_grad.ptr(),
                self.loss_values.ptr(),
                self.predictions.ptr(),
                self.logits.ptr(),
                self.labels.ptr(),
                self.config.vocab as u32,
                rows as u32,
                normalizer as f32,
            )?;
        }
        self.last = Some((input, scores, batch, length));
        Ok(())
    }

    fn inner(&mut self, input: Pair, batch: usize, length: usize) -> Result<Pair> {
        let rows = batch * length;
        let p = &self.parameters;
        let width = self.shape.width;
        let heads = self.shape.heads;
        let head_width = width / heads;
        let alpha = p.alpha();
        assert_eq!(input.len, rows * 2 * width);
        let ctx = Context {
            cuda: &self.cuda,
            kernel: &self.kernel,
            gemm: &self.gemm,
            workspace: self.workspace.ptr(),
            workspace_len: self.workspace.len(),
            multiprocessors: self.cuda.multiprocessors() as usize,
            scores: &self.scores,
            rotations: self.rotations.ptr(),
            alpha,
            length,
            heads,
            head_width,
        };
        let tape = &mut self.tape;
        // SAFETY: the table has three reals per frequency and position.
        unsafe {
            self.kernel.rope_table(
                grid(length * head_width),
                ctx.rotations,
                alpha.x,
                head_width as u32,
                (length * head_width) as u32,
                -10_000_f32.ln() / head_width as f32,
            )?;
        }
        let mut x = tape.rms(&ctx, input, width)?;
        let scale = (2.0 * (self.shape.layers * self.repeat) as f32)
            .sqrt()
            .recip();
        for _ in 0..self.repeat {
            for layer in &p.layers {
                let attention = Sublayer::read(tape, &ctx, p, layer, 0, x, width)?;
                let n = attention.input;
                let q = layer.q.project(tape, &ctx, p, n, rows)?;
                let k = layer.k.project(tape, &ctx, p, n, rows)?;
                let v = layer.v.project(tape, &ctx, p, n, rows)?;
                let q = tape.rms(&ctx, q, head_width)?;
                let k = tape.rms(&ctx, k, head_width)?;
                let q = tape.rotate(&ctx, q)?;
                let k = tape.rotate(&ctx, k)?;
                let q = tape
                    .heads(&ctx, q, false)?
                    .matrix(batch * heads, length, head_width);
                let k = tape
                    .heads(&ctx, k, false)?
                    .matrix(batch * heads, length, head_width);
                let v = tape
                    .heads(&ctx, v, false)?
                    .matrix(batch * heads, length, head_width);
                let score = tape.mm(&ctx, q.conjugate(), k.transpose())?;
                let score = tape.scores(&ctx, score, p.pair(layer.score, 2))?;
                let a = tape.mm(&ctx, score.matrix(batch * heads, length, length), v)?;
                let a = tape.heads(&ctx, a, true)?;
                let a = layer.o.project(tape, &ctx, p, a, rows)?;
                let a = tape.rms(&ctx, a, width)?;
                x = attention.write(tape, &ctx, a, scale)?;

                let ffn = Sublayer::read(tape, &ctx, p, layer, 1, x, width)?;
                let n = ffn.input;
                let g = layer.gate.project(tape, &ctx, p, n, rows)?;
                let u = layer.up.project(tape, &ctx, p, n, rows)?;
                let f = tape.product(&ctx, g, u)?;
                let f = layer.down.project(tape, &ctx, p, f, rows)?;
                let f = tape.rms(&ctx, f, width)?;
                x = ffn.write(tape, &ctx, f, scale)?;
            }
        }
        tape.rms(&ctx, x, width)
    }

    fn readout(&mut self, hidden: Pair, batch: usize, length: usize) -> Result<Pair> {
        let rows = batch * length;
        let width = self.shape.width;
        let heads = self.shape.heads;
        let ctx = Context {
            cuda: &self.cuda,
            kernel: &self.kernel,
            gemm: &self.gemm,
            workspace: self.workspace.ptr(),
            workspace_len: self.workspace.len(),
            multiprocessors: self.cuda.multiprocessors() as usize,
            scores: &self.scores,
            rotations: self.rotations.ptr(),
            alpha: self.parameters.alpha(),
            length,
            heads,
            head_width: width / heads,
        };
        let h = hidden.matrix(1, rows, width);
        let classifier = self
            .parameters
            .embedding()
            .matrix(1, self.config.vocab, width)
            .transpose()
            .conjugate();
        let scores = self.tape.mm(&ctx, h, classifier)?;
        let count = rows * self.config.vocab;
        // SAFETY: each row of scores covers the entire vocabulary.
        unsafe {
            self.kernel.real_readout::<false>(
                grid(count),
                self.logits.ptr(),
                scores.x,
                count as u32,
            )?;
        }
        Ok(scores)
    }

    pub fn losses(&self, rows: usize) -> Result<Vec<f32>> {
        Ok(self
            .loss_values
            .part(..rows)
            .ok_or("loss capacity")?
            .read()?)
    }

    fn backward(&mut self) -> Result<()> {
        self.zero_gradient()?;
        self.add_backward()
    }

    /// Clears the parameter gradient before a full-batch accumulation.
    pub fn zero_gradient(&mut self) -> Result<()> {
        let p = &self.parameters;
        // SAFETY: the launch covers the parameter gradient buffer.
        unsafe {
            self.kernel
                .clear(grid(p.len()), p.grad.ptr(), p.len() as u32)?;
        }
        Ok(())
    }

    /// Adds one microbatch's part of the mean gradient over `total` labeled
    /// positions. Over a partition of the training split the parts sum to
    /// the gradient of the mean loss (`accumulated_batchGradient`).
    pub fn accumulate(
        &mut self,
        tokens: &[u32],
        labels: &[u32],
        batch: usize,
        length: usize,
        total: usize,
    ) -> Result<()> {
        self.forward_normalized(tokens, labels, batch, length, Some(total))?;
        self.add_backward()
    }

    /// One update of `fullGradientMomentumRun` from the accumulated data
    /// gradient: adds `PENALTY * x`, then the clipped momentum step.
    pub fn full_step(&mut self, rate: f32) -> Result<()> {
        self.penalized_step(PENALTY, rate)
    }

    fn penalized_step(&mut self, penalty: f32, rate: f32) -> Result<()> {
        self.parameters.add_penalty(&self.kernel, penalty)?;
        self.parameters.update(&self.optimizer, &self.kernel, rate)
    }

    /// Backpropagates the last forward pass and adds its parameter gradient.
    /// Every parameter adjoint accumulates: matrix products and gates add,
    /// and the LeRoPE and score adjoints use atomic additions.
    fn add_backward(&mut self) -> Result<()> {
        let (input, output, batch, length) =
            self.last.take().ok_or("forward required before update")?;
        let width = self.shape.width;
        let heads = self.shape.heads;
        let ctx = Context {
            cuda: &self.cuda,
            kernel: &self.kernel,
            gemm: &self.gemm,
            workspace: self.workspace.ptr(),
            workspace_len: self.workspace.len(),
            multiprocessors: self.cuda.multiprocessors() as usize,
            scores: &self.scores,
            rotations: self.rotations.ptr(),
            alpha: self.parameters.alpha(),
            length,
            heads,
            head_width: width / heads,
        };
        self.tape.clear_gradients(&self.kernel)?;
        // SAFETY: this launch uses the logit buffer extent.
        unsafe {
            let count = output.len / 2;
            self.kernel.real_readout::<true>(
                grid(count),
                output.dx,
                self.logit_grad.ptr(),
                count as u32,
            )?;
        }
        self.tape.backward(&ctx)?;
        // SAFETY: input IDs were checked and this gradient adds to the tied readout's.
        unsafe {
            let e = self.parameters.embedding();
            self.embed.embed_bwd(
                grid(e.len),
                e.dx,
                input.dx,
                self.inputs.ptr(),
                (2 * width) as u32,
                self.config.vocab as u32,
                (batch * length) as u32,
            )?;
        }
        Ok(())
    }

    /// The matrix-product operations of the last forward pass and its
    /// backward pass.
    pub fn product_flops(&self) -> f64 {
        self.tape.product_flops()
    }

    pub fn state(&self) -> Result<Vec<f32>> {
        self.parameters.state()
    }

    pub fn restore_state(&mut self, state: &[f32]) -> Result<()> {
        self.last = None;
        self.parameters.restore_state(state)
    }

    pub fn weights(&self) -> Result<Vec<f32>> {
        Ok(self.parameters.value.read()?)
    }

    pub fn restore(&mut self, weights: &[f32]) -> Result<()> {
        if weights.len() != self.parameters.len() || weights.iter().any(|x| !x.is_finite()) {
            return Err("invalid Argand checkpoint".into());
        }
        self.last = None;
        Ok(self.parameters.value.write(weights)?)
    }
}

pub fn check() -> Result<()> {
    let cfg = Shape {
        width: 16,
        heads: 2,
        layers: 2,
        vocab: 6,
        max_length: 3,
        max_batch: 2,
    };
    let mut engine = Engine::new(cfg, 2, 1)?;
    let products = Context {
        cuda: &engine.cuda,
        kernel: &engine.kernel,
        gemm: &engine.gemm,
        workspace: engine.workspace.ptr(),
        workspace_len: engine.workspace.len(),
        multiprocessors: engine.cuda.multiprocessors() as usize,
        scores: &engine.scores,
        rotations: engine.rotations.ptr(),
        alpha: engine.parameters.alpha(),
        length: 1,
        heads: 1,
        head_width: 1,
    }
    .check_products()?;
    println!(
        "Tiled complex products equal the 16 x 16 kernel's with k whole, and agree within {products:.1e} of the largest value with k split"
    );
    let mut initial = engine.weights()?;
    let width = engine.shape.width;
    if engine
        .parameters
        .layers
        .iter()
        .any(|layer| initial[layer.score..layer.score + 2] != SCORE_INIT)
    {
        return Err("score multiplier has the wrong initialization".into());
    }
    let mut tied = vec![0.0; initial.len()];
    tied[2 * 16] = 0.125;
    tied[2 * 2 * 16] = 0.25;
    engine.restore(&tied)?;
    engine.forward(&[1], &[2], 1, 1)?;
    let logits = engine.logits.part(..6).ok_or("logit capacity")?.read()?;
    let loss = engine.losses(1)?[0];
    if engine.predictions.read()?[0] != 2
        || !loss.is_finite()
        || (logits[2] - 0.5).abs() > 1e-5
        || (logits[1] - 0.25).abs() > 1e-5
    {
        return Err("the tied classifier failed to predict the unseen token".into());
    }

    // Each layer appends its attention and FFN write gates, then its two skip
    // gates, after all other coordinates. All start at zero.
    let gates = 8 * width * engine.shape.layers;
    let first_gate = initial.len() - gates;
    let offsets: Vec<_> = engine
        .parameters
        .layers
        .iter()
        .flat_map(|layer| layer.write_gates.into_iter().chain(layer.skip_gates))
        .collect();
    if initial[first_gate..].iter().any(|&v| v != 0.0)
        || offsets
            .iter()
            .enumerate()
            .any(|(j, &at)| at != first_gate + 2 * width * j)
    {
        return Err("the gate vectors are not the trailing zero coordinates".into());
    }
    // Exercise the input and parameter adjoints away from the identity.
    for &at in &offsets {
        for (j, value) in initial[at..at + 2 * width].iter_mut().enumerate() {
            *value = 0.01 * ((j % 7) as f32 - 3.0);
        }
    }
    engine.restore(&initial)?;
    let tokens = [1, 2, 3, 3, 1, 0];
    let labels = [2, 3, 4, u32::MAX, 5, 2];
    engine.forward(&tokens, &labels, 2, 3)?;
    let before = engine.logits.read()?;
    let dense: Vec<_> = tokens
        .iter()
        .flat_map(|&id| {
            initial[id as usize * 32..(id as usize + 1) * 32]
                .iter()
                .copied()
        })
        .collect();
    let dense_input = engine.cuda.upload(&dense)?;
    let dense_grad = engine.cuda.alloc(dense.len())?;
    engine.tape.reset();
    let hidden = engine.inner(
        Pair {
            x: dense_input.ptr(),
            dx: dense_grad.ptr(),
            len: dense.len(),
        },
        2,
        3,
    )?;
    engine.readout(hidden, 2, 3)?;
    if engine.logits.read()? != before {
        return Err("dense embedding input disagrees with the token lookup".into());
    }
    engine.forward(&[1, 2, 5, 3, 1, 5], &[0; 6], 2, 3)?;
    let after = engine.logits.read()?;
    for i in (0..12).chain(18..30) {
        if (before[i] - after[i]).abs() > 2e-6 {
            return Err("future inputs or labels changed a prefix logit".into());
        }
    }

    engine.forward(&tokens, &labels, 2, 3)?;
    engine.backward()?;
    let gradient = engine.parameters.grad.read()?;
    for &at in &offsets {
        let g = &gradient[at..at + 2 * width];
        if (0..2).any(|coord| g[coord..].iter().step_by(2).all(|x| x.abs() < 1e-6)) {
            return Err("missing real or imaginary token-gate gradient".into());
        }
    }
    if !gradient.iter().all(|g| g.is_finite())
        || !gradient[2 * 4 * 16..2 * 5 * 16]
            .iter()
            .any(|g| g.abs() > 1e-6)
    {
        return Err("missing gradient for a vocabulary row absent from lookup".into());
    }
    if !engine
        .parameters
        .layers
        .iter()
        .any(|layer| gradient[layer.score + 1].abs() > 1e-6)
    {
        return Err("missing gradient for the imaginary part of the score multiplier".into());
    }
    let mut rng = Random::new(100);
    for trial in 0..5 {
        let mut direction: Vec<_> = initial.iter().map(|_| rng.gauss(0.0, 1.0) as f32).collect();
        if trial >= 3 {
            // Trial 3 probes the write gates alone. Trial 4 probes the skip gates alone.
            direction[..first_gate].fill(0.0);
            for layer in &engine.parameters.layers {
                let excluded = if trial == 3 {
                    layer.skip_gates
                } else {
                    layer.write_gates
                };
                for at in excluded {
                    direction[at..at + 2 * width].fill(0.0);
                }
            }
        }
        let delta = 0.0002;
        let mut losses = [0.0; 2];
        for (loss, sign) in losses.iter_mut().zip([-1.0, 1.0]) {
            let weights: Vec<_> = initial
                .iter()
                .zip(&direction)
                .map(|(x, d)| x + d * delta * sign)
                .collect();
            engine.restore(&weights)?;
            engine.forward(&tokens, &labels, 2, 3)?;
            *loss = engine.losses(6)?.iter().map(|&x| f64::from(x)).sum::<f64>() / 5.0;
        }
        let finite = (losses[1] - losses[0]) / f64::from(2.0 * delta);
        let analytic: f64 = gradient
            .iter()
            .zip(&direction)
            .map(|(&g, &d)| f64::from(g) * f64::from(d))
            .sum();
        println!("direction {trial}: analytic={analytic:.6} finite_difference={finite:.6}");
        if !finite.is_finite() || (finite - analytic).abs() > 0.02 * (1.0 + finite.abs()) {
            return Err("complex decoder GPU backward disagrees with finite differences".into());
        }
    }
    // The two examples hold three and two of the five labels. Each part
    // divides by all five, so the parts sum to the batch gradient.
    engine.restore(&initial)?;
    engine.forward(&tokens, &labels, 2, 3)?;
    engine.backward()?;
    let whole = engine.parameters.grad.read()?;
    engine.zero_gradient()?;
    for rows in [0..3, 3..6] {
        engine.accumulate(&tokens[rows.clone()], &labels[rows], 1, 3, 5)?;
    }
    let parts = engine.parameters.grad.read()?;
    let largest = whole.iter().fold(0.0_f32, |m, g| m.max(g.abs()));
    let gap = parts
        .iter()
        .zip(&whole)
        .fold(0.0_f32, |m, (&a, &b)| m.max((a - b).abs()));
    if parts.len() != whole.len()
        || !parts.iter().all(|g| g.is_finite())
        || gap > 1e-5 * (1.0 + largest)
    {
        return Err("accumulated microbatch gradients disagree with the batch gradient".into());
    }
    println!(
        "Label-weighted microbatch gradients sum to the batch gradient within {gap:.1e} of {largest:.3}"
    );
    // The penalty enters the gradient once, before clipping. A zero rate
    // keeps the weights and sets the velocity to the clipped sum. A penalty
    // far above PENALTY makes a missing or doubled term visible.
    let penalty = 0.5_f32;
    let before = engine.state()?;
    engine.penalized_step(penalty, 0.0)?;
    let after = engine.state()?;
    let count = parts.len();
    let sum: Vec<f64> = parts
        .iter()
        .zip(&initial)
        .map(|(&g, &x)| f64::from(g) + f64::from(penalty) * f64::from(x))
        .collect();
    let norm = sum.iter().map(|g| g * g).sum::<f64>().sqrt();
    let scale = norm.max(1.0);
    if after[..count] != before[..count]
        || after[count..]
            .iter()
            .zip(&before[count..])
            .zip(&sum)
            .any(|((&v, &old), &g)| {
                (f64::from(v) - (0.9 * f64::from(old) + g / scale)).abs() > 2e-6
            })
    {
        return Err("penalized full-batch step disagrees with the reference".into());
    }
    println!("Full-batch penalty enters once before clipping at gradient norm {norm:.4}");
    engine
        .parameters
        .check_sgd_momentum(&engine.optimizer, &engine.kernel)?;
    println!("{MODEL} vocabulary prediction, prefix causality and tied adjoints verified");
    Ok(())
}

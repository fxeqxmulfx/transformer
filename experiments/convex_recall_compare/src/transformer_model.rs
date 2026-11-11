//! Copied from `experiments/grok_division/src/model.rs` at 891e4525.
//! The two attention scratch arrays are sized from the context instead of capped
//! at sixteen. Forward, backward, activations, precision, and loss are unchanged.
//!
//! The model in `f32`, a chunk of examples at a time: every product of a matrix of activations
//! by a matrix of weights is one call of the blocked kernel. The second block computes only the
//! supervised positions, which is every output the loss reads; positions past the last
//! supervised one are not computed at all, since the causal mask hides them from the others.
//! The tests hold the logits and the gradient to the reference model.

use crate::gemm::{Gemm, transpose};
use crate::layout::{Dims, LN_EPS};

/// The weights of the linear maps transposed, `out` rows of `in`, for the backward products.
pub struct Transposed {
    blocks: Vec<[Vec<f32>; 5]>,
    wout: Vec<f32>,
}

impl Transposed {
    /// Room for the transposes of a model of `dims`.
    pub fn new(dims: Dims) -> Transposed {
        let (d, dff) = (dims.d, dims.dff);
        let block = || {
            [vec![0.0; d * d], vec![0.0; 2 * d * d], vec![0.0; d * d], vec![0.0; d * dff], vec![0.0; dff * d]]
        };
        Transposed { blocks: (0..dims.layers).map(|_| block()).collect(), wout: vec![0.0; d * dims.vocab] }
    }

    /// Transposes the weights in `p`.
    pub fn update(&mut self, dims: Dims, p: &[f32]) {
        let (d, dff) = (dims.d, dims.dff);
        for (l, [wq, wkv, wo, w1, w2]) in self.blocks.iter_mut().enumerate() {
            let at = dims.block(l);
            transpose(wq, &p[at.wq..at.wq + d * d], d, d);
            transpose(wkv, &p[at.wkv..at.wkv + 2 * d * d], d, 2 * d);
            transpose(wo, &p[at.wo..at.wo + d * d], d, d);
            transpose(w1, &p[at.w1..at.w1 + d * dff], d, dff);
            transpose(w2, &p[at.w2..at.w2 + dff * d], dff, d);
        }
        transpose(&mut self.wout, &p[dims.wout()..dims.wout() + d * dims.vocab], d, dims.vocab);
    }
}

/// What a block keeps of a chunk for its backward pass. Rows of queries: an example's query
/// positions in order; rows of keys and values: every position read.
#[derive(Default)]
struct Trace {
    xs: Vec<f32>,
    q: Vec<f32>,
    kv: Vec<f32>,
    att: Vec<f32>,
    o: Vec<f32>,
    hat1: Vec<f32>,
    isig1: Vec<f32>,
    y1: Vec<f32>,
    r: Vec<f32>,
    hat2: Vec<f32>,
    isig2: Vec<f32>,
    out: Vec<f32>,
}

/// A thread's memory: the kernel's scratch, the activations of its chunk, and its gradient.
#[derive(Default)]
pub struct Work {
    gemm: Gemm,
    x0: Vec<f32>,
    traces: [Trace; 2],
    /// The logits of the chunk: an example's supervised positions in order, `vocab` floats each.
    pub logits: Vec<f32>,
    /// The gradient of the chunk's share of the loss, in every parameter.
    pub grad: Vec<f32>,
    t: Vec<f32>,
    u: Vec<f32>,
    v: Vec<f32>,
    dq: Vec<f32>,
    dkv: Vec<f32>,
    dx: Vec<f32>,
    dout: Vec<f32>,
}

/// The model of one step: its sizes, parameters, positional encoding and supervised positions.
pub struct Model<'a> {
    /// The sizes; two layers.
    pub dims: Dims,
    /// The flat parameters.
    pub p: &'a [f32],
    /// The positional encoding, `ctx` rows of `d`.
    pub pe: &'a [f32],
    /// The transposed weights of `p`; read by [`Model::backward`] only.
    pub wt: &'a Transposed,
    /// The positions whose outputs the loss reads, increasing.
    pub sup: &'a [usize],
}

/// Layer normalization of the row `z` in place, with gain `g` and bias `b`: the normalized row
/// goes to `hat`, and the reciprocal of the floored standard deviation is returned. The mean
/// and the variance are accumulated in doubles.
fn normalize(z: &mut [f32], hat: &mut [f32], g: &[f32], b: &[f32]) -> f32 {
    let n = z.len();
    let mean = sum4(n, |i| z[i] as f64) / n as f64;
    let var = sum4(n, |i| (z[i] as f64 - mean) * (z[i] as f64 - mean)) / n as f64;
    let isig = 1.0 / (var + LN_EPS).sqrt();
    for i in 0..n {
        let h = ((z[i] as f64 - mean) * isig) as f32;
        hat[i] = h;
        z[i] = h * g[i] + b[i];
    }
    isig as f32
}

/// The gradient through a layer normalization, in place: `dy` of the output becomes that of the
/// input; the gain's and the bias's are added to `gg` and `gb`.
fn normalize_back(dy: &mut [f32], hat: &[f32], isig: f32, g: &[f32], gg: &mut [f64], gb: &mut [f64]) {
    let n = dy.len();
    for i in 0..n {
        gg[i] += dy[i] as f64 * hat[i] as f64;
        gb[i] += dy[i] as f64;
        dy[i] *= g[i];
    }
    let mean = (sum4(n, |i| dy[i] as f64) / n as f64) as f32;
    let along = (sum4(n, |i| dy[i] as f64 * hat[i] as f64) / n as f64) as f32;
    for i in 0..n {
        dy[i] = (dy[i] - mean - hat[i] * along) * isig;
    }
}

/// The sum of `f(i)` over `i < n` in doubles, as four interleaved partial sums added as
/// `(s0 + s2) + (s1 + s3)`, then the last `n mod 4` terms in order.
fn sum4(n: usize, f: impl Fn(usize) -> f64) -> f64 {
    let mut s = [0f64; 4];
    for i in (0..n - n % 4).step_by(4) {
        for (lane, part) in s.iter_mut().enumerate() {
            *part += f(i + lane);
        }
    }
    (n - n % 4..n).fold((s[0] + s[2]) + (s[1] + s[3]), |total, i| total + f(i))
}

/// The dot product of two rows of equal length in `f32`, as eight interleaved partial sums added
/// as `((s0 + s4) + (s2 + s6)) + ((s1 + s5) + (s3 + s7))`, then the last terms in order.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut s = [0f32; 8];
    let (rows_a, rows_b) = (a.chunks_exact(8), b.chunks_exact(8));
    let (rest_a, rest_b) = (rows_a.remainder(), rows_b.remainder());
    for (x, y) in rows_a.zip(rows_b) {
        for lane in 0..8 {
            s[lane] += x[lane] * y[lane];
        }
    }
    let total = ((s[0] + s[4]) + (s[2] + s[6])) + ((s[1] + s[5]) + (s[3] + s[7]));
    rest_a.iter().zip(rest_b).fold(total, |total, (x, y)| total + x * y)
}

impl Model<'_> {
    /// Positions read: up to the last supervised one.
    fn read(&self) -> usize {
        self.sup[self.sup.len() - 1] + 1
    }

    /// The query positions of block `l`: all those read in the first, the supervised in the last.
    fn queries(&self, l: usize) -> Vec<usize> {
        if l == 0 { (0..self.read()).collect() } else { self.sup.to_vec() }
    }

    /// One block on a chunk of `n` examples: from `x`, a row for every position read, to
    /// `tr.out`, a row for every query position.
    fn block_forward(&self, gemm: &mut Gemm, x: &[f32], tr: &mut Trace, l: usize, n: usize) {
        let (d, heads, dk, dff, t_len) =
            (self.dims.d, self.dims.heads, self.dims.dk(), self.dims.dff, self.read());
        let (at, qpos) = (self.dims.block(l), self.queries(l));
        let (nq, rows) = (qpos.len(), n * t_len);
        let qrows = n * nq;
        tr.xs.resize(qrows * d, 0.0);
        for b in 0..n {
            for (qi, &t) in qpos.iter().enumerate() {
                tr.xs[(b * nq + qi) * d..][..d].copy_from_slice(&x[(b * t_len + t) * d..][..d]);
            }
        }
        tr.q.resize(qrows * d, 0.0);
        gemm.mul(&mut tr.q, &tr.xs, &self.p[at.wq..at.wq + d * d], (qrows, d, d));
        tr.kv.resize(rows * 2 * d, 0.0);
        gemm.mul(&mut tr.kv, x, &self.p[at.wkv..at.wkv + 2 * d * d], (rows, 2 * d, d));

        tr.att.clear();
        tr.att.resize(qrows * heads * t_len, 0.0);
        tr.o.resize(qrows * d, 0.0);
        let root = (dk as f32).sqrt();
        let mut s = vec![0f32; t_len];
        for b in 0..n {
            for (qi, &t) in qpos.iter().enumerate() {
                let qr = b * nq + qi;
                for h in 0..heads {
                    let qv = &tr.q[qr * d + h * dk..][..dk];
                    let mut top = f32::NEG_INFINITY;
                    for j in 0..=t {
                        s[j] = dot(qv, &tr.kv[(b * t_len + j) * 2 * d + h * dk..][..dk]) / root;
                        top = top.max(s[j]);
                    }
                    let mut total = 0f32;
                    for sj in s.iter_mut().take(t + 1) {
                        *sj = (*sj - top).exp();
                        total += *sj;
                    }
                    let orow = &mut tr.o[qr * d + h * dk..][..dk];
                    orow.fill(0.0);
                    for j in 0..=t {
                        let a = s[j] / total;
                        tr.att[(qr * heads + h) * t_len + j] = a;
                        let vv = &tr.kv[(b * t_len + j) * 2 * d + d + h * dk..][..dk];
                        for r in 0..dk {
                            orow[r] += a * vv[r];
                        }
                    }
                }
            }
        }

        tr.y1.resize(qrows * d, 0.0);
        gemm.mul(&mut tr.y1, &tr.o, &self.p[at.wo..at.wo + d * d], (qrows, d, d));
        tr.hat1.resize(qrows * d, 0.0);
        tr.isig1.resize(qrows, 0.0);
        let (g1, b1) = (&self.p[at.g1..at.g1 + d], &self.p[at.b1..at.b1 + d]);
        for r in 0..qrows {
            let z = &mut tr.y1[r * d..][..d];
            for i in 0..d {
                z[i] += tr.xs[r * d + i];
            }
            tr.isig1[r] = normalize(z, &mut tr.hat1[r * d..][..d], g1, b1);
        }
        tr.r.resize(qrows * dff, 0.0);
        gemm.mul(&mut tr.r, &tr.y1, &self.p[at.w1..at.w1 + d * dff], (qrows, dff, d));
        for v in tr.r.iter_mut() {
            *v = v.max(0.0);
        }
        tr.out.resize(qrows * d, 0.0);
        gemm.mul(&mut tr.out, &tr.r, &self.p[at.w2..at.w2 + dff * d], (qrows, d, dff));
        tr.hat2.resize(qrows * d, 0.0);
        tr.isig2.resize(qrows, 0.0);
        let (g2, b2) = (&self.p[at.g2..at.g2 + d], &self.p[at.b2..at.b2 + d]);
        for r in 0..qrows {
            let z = &mut tr.out[r * d..][..d];
            for i in 0..d {
                z[i] += tr.y1[r * d + i];
            }
            tr.isig2[r] = normalize(z, &mut tr.hat2[r * d..][..d], g2, b2);
        }
    }

    /// The logits of a chunk into `w.logits`: `tokens` holds `ctx` tokens for each example.
    pub fn forward(&self, w: &mut Work, tokens: &[usize]) {
        let (d, ctx, t_len, v) = (self.dims.d, self.dims.ctx, self.read(), self.dims.vocab);
        assert_eq!(self.dims.layers, 2, "the fast model has two blocks");
        assert_eq!(tokens.len() % ctx, 0);
        let n = tokens.len() / ctx;
        w.x0.resize(n * t_len * d, 0.0);
        for b in 0..n {
            for t in 0..t_len {
                let row = &mut w.x0[(b * t_len + t) * d..][..d];
                let emb = &self.p[self.dims.emb() + tokens[b * ctx + t] * d..][..d];
                for i in 0..d {
                    row[i] = emb[i] + self.pe[t * d + i];
                }
            }
        }
        let [tr0, tr1] = &mut w.traces;
        self.block_forward(&mut w.gemm, &w.x0, tr0, 0, n);
        self.block_forward(&mut w.gemm, &tr0.out, tr1, 1, n);
        let srows = n * self.sup.len();
        w.logits.resize(srows * v, 0.0);
        w.gemm.mul(
            &mut w.logits,
            &tr1.out,
            &self.p[self.dims.wout()..self.dims.wout() + d * v],
            (srows, v, d),
        );
    }

    /// The backward pass of one block: `w.dout`, the gradient of its output, becomes `w.dx`,
    /// that of its input `x`; the gradients of its parameters go to `w.grad`.
    fn block_backward(&self, w: &mut Work, l: usize, n: usize) {
        let (d, heads, dk, dff, t_len) =
            (self.dims.d, self.dims.heads, self.dims.dk(), self.dims.dff, self.read());
        let (at, qpos, [wq_t, wkv_t, wo_t, w1_t, w2_t]) =
            (self.dims.block(l), self.queries(l), &self.wt.blocks[l]);
        let (nq, rows) = (qpos.len(), n * t_len);
        let qrows = n * nq;
        let Work { gemm, x0, traces, grad, t, u, v, dq, dkv, dx, dout, .. } = w;
        let (before, after) = traces.split_at_mut(l);
        let tr = &mut after[0];
        let x: &[f32] = if l == 0 { &x0[..] } else { &before[l - 1].out };
        let (mut gg, mut gb) = (vec![0f64; d], vec![0f64; d]);

        for r in 0..qrows {
            let g2 = &self.p[at.g2..at.g2 + d];
            normalize_back(
                &mut dout[r * d..][..d],
                &tr.hat2[r * d..][..d],
                tr.isig2[r],
                g2,
                &mut gg,
                &mut gb,
            );
        }
        for i in 0..d {
            grad[at.g2 + i] = gg[i] as f32;
            grad[at.b2 + i] = gb[i] as f32;
        }
        t.resize(dff * qrows, 0.0);
        transpose(t, &tr.r, qrows, dff);
        gemm.mul(&mut grad[at.w2..at.w2 + dff * d], t, dout, (dff, d, qrows));
        u.resize(qrows * dff, 0.0);
        gemm.mul(u, dout, w2_t, (qrows, dff, d));
        for (du, &r) in u.iter_mut().zip(tr.r.iter()) {
            if r <= 0.0 {
                *du = 0.0;
            }
        }
        t.resize(d * qrows, 0.0);
        transpose(t, &tr.y1, qrows, d);
        gemm.mul(&mut grad[at.w1..at.w1 + d * dff], t, u, (d, dff, qrows));
        v.resize(qrows * d, 0.0);
        gemm.mul(v, u, w1_t, (qrows, d, dff));
        for (a, b) in dout.iter_mut().zip(v.iter()) {
            *a += *b;
        }

        gg.fill(0.0);
        gb.fill(0.0);
        for r in 0..qrows {
            let g1 = &self.p[at.g1..at.g1 + d];
            normalize_back(
                &mut dout[r * d..][..d],
                &tr.hat1[r * d..][..d],
                tr.isig1[r],
                g1,
                &mut gg,
                &mut gb,
            );
        }
        for i in 0..d {
            grad[at.g1 + i] = gg[i] as f32;
            grad[at.b1 + i] = gb[i] as f32;
        }
        transpose(t, &tr.o, qrows, d);
        gemm.mul(&mut grad[at.wo..at.wo + d * d], t, dout, (d, d, qrows));
        gemm.mul(v, dout, wo_t, (qrows, d, d));

        dq.clear();
        dq.resize(qrows * d, 0.0);
        dkv.clear();
        dkv.resize(rows * 2 * d, 0.0);
        let root = (dk as f32).sqrt();
        let mut da = vec![0f32; t_len];
        for b in 0..n {
            for (qi, &tq) in qpos.iter().enumerate() {
                let qr = b * nq + qi;
                for h in 0..heads {
                    let a = &tr.att[(qr * heads + h) * t_len..][..t_len];
                    let doh = &v[qr * d + h * dk..][..dk];
                    let mut mean = 0f32;
                    for j in 0..=tq {
                        let base = (b * t_len + j) * 2 * d + d + h * dk;
                        da[j] = dot(doh, &tr.kv[base..][..dk]);
                        mean += a[j] * da[j];
                        for r in 0..dk {
                            dkv[base + r] += a[j] * doh[r];
                        }
                    }
                    let qv = &tr.q[qr * d + h * dk..][..dk];
                    for j in 0..=tq {
                        let ds = a[j] * (da[j] - mean) / root;
                        let base = (b * t_len + j) * 2 * d + h * dk;
                        for r in 0..dk {
                            dq[qr * d + h * dk + r] += ds * tr.kv[base + r];
                            dkv[base + r] += ds * qv[r];
                        }
                    }
                }
            }
        }
        transpose(t, &tr.xs, qrows, d);
        gemm.mul(&mut grad[at.wq..at.wq + d * d], t, dq, (d, d, qrows));
        gemm.mul(v, dq, wq_t, (qrows, d, d));
        for (a, b) in dout.iter_mut().zip(v.iter()) {
            *a += *b;
        }
        t.resize(d * rows, 0.0);
        transpose(t, x, rows, d);
        gemm.mul(&mut grad[at.wkv..at.wkv + 2 * d * d], t, dkv, (d, 2 * d, rows));
        dx.resize(rows * d, 0.0);
        gemm.mul(dx, dkv, wkv_t, (rows, d, 2 * d));
        for b in 0..n {
            for (qi, &tq) in qpos.iter().enumerate() {
                let (to, from) = (&mut dx[(b * t_len + tq) * d..][..d], &dout[(b * nq + qi) * d..][..d]);
                for i in 0..d {
                    to[i] += from[i];
                }
            }
        }
    }

    /// After [`Model::forward`] on the same chunk: the gradient of `scale` times the sum of the
    /// chunk's cross-entropies into `w.grad`, and that sum for each supervised position.
    /// `targets` holds a target for each supervised position of each example.
    pub fn backward(&self, w: &mut Work, tokens: &[usize], targets: &[usize], scale: f32) -> Vec<f64> {
        let (d, ctx, t_len, v, ns) =
            (self.dims.d, self.dims.ctx, self.read(), self.dims.vocab, self.sup.len());
        let n = tokens.len() / ctx;
        let srows = n * ns;
        assert_eq!(targets.len(), srows);
        // Every tensor but the embedding is written whole below; the embedding is added to.
        w.grad.resize(self.dims.len(), 0.0);
        w.grad[self.dims.emb()..self.dims.emb() + v * d].fill(0.0);
        let mut sums = vec![0f64; ns];
        for r in 0..srows {
            sums[r % ns] += loss_gradient(&mut w.logits[r * v..][..v], targets[r], scale);
        }
        w.t.resize(d * srows, 0.0);
        transpose(&mut w.t, &w.traces[1].out, srows, d);
        let at = self.dims.wout();
        w.gemm.mul(&mut w.grad[at..at + d * v], &w.t, &w.logits, (d, v, srows));
        w.dout.resize(srows * d, 0.0);
        w.gemm.mul(&mut w.dout, &w.logits, &self.wt.wout, (srows, d, v));
        self.block_backward(w, 1, n);
        std::mem::swap(&mut w.dout, &mut w.dx);
        self.block_backward(w, 0, n);
        for b in 0..n {
            for t in 0..t_len {
                let to = &mut w.grad[self.dims.emb() + tokens[b * ctx + t] * d..][..d];
                let from = &w.dx[(b * t_len + t) * d..][..d];
                for i in 0..d {
                    to[i] += from[i];
                }
            }
        }
        sums
    }
}

/// A row of logits becomes the gradient of `weight` times its cross-entropy against `target`,
/// in `f32` as PyTorch's `F.cross_entropy` computes it; the cross-entropy is returned.
///
/// `log_softmax` is `x - max - ln(sum exp(x - max))`, the loss is `-log_softmax[target]`, and
/// the backward pass of `log_softmax` is `grad_output - exp(output) * sum(grad_output)` with
/// `grad_output = -weight` at the target and 0 elsewhere. Once the other classes' probabilities
/// sum to less than half a unit in the last place of 1, the sum rounds to 1, the loss to 0, and
/// the target's entry `-weight + exp(0) * weight` is exactly 0, while every other class keeps
/// its entry `exp(x - max) * weight`: single precision's softmax collapse, which the source's
/// runs have and which arithmetic in doubles would hide.
pub fn loss_gradient(row: &mut [f32], target: usize, weight: f32) -> f64 {
    let top = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut total = 0f32;
    for &value in row.iter() {
        total += (value - top).exp();
    }
    let log_total = total.ln();
    let mut loss = 0f64;
    for (c, value) in row.iter_mut().enumerate() {
        let log_probability = (*value - top) - log_total;
        let from_all = log_probability.exp() * weight;
        if c == target {
            loss = -(log_probability as f64);
            *value = -weight + from_all;
        } else {
            *value = from_all;
        }
    }
    loss
}

/// The cross-entropy of a row of logits against `target`, and the probabilities, in doubles:
/// for the evaluations, which report it and take no gradient.
pub fn softmax(row: &[f32], target: usize) -> (f64, Vec<f64>) {
    let top = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut probabilities: Vec<f64> = row.iter().map(|&v| (v as f64 - top).exp()).collect();
    let total: f64 = probabilities.iter().sum();
    probabilities.iter_mut().for_each(|p| *p /= total);
    (total.ln() + top - row[target] as f64, probabilities)
}

/// The class a row of logits predicts, the first of the largest, and the margin of `target`:
/// its logit less the largest of the others.
pub fn decide(row: &[f32], target: usize) -> (usize, f32) {
    let (mut best, mut rival) = (0, f32::NEG_INFINITY);
    for (c, &value) in row.iter().enumerate() {
        if value > row[best] {
            best = c;
        }
        if c != target && value > rival {
            rival = value;
        }
    }
    (best, row[target] - rival)
}

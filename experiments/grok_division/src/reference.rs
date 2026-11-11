//! The model in doubles, one example and one float at a time, straight from
//! openai/grok `grok/transformer.py`: what the fast model is held to. Its own gradient is held
//! to finite differences of its forward pass by the tests.

use crate::layout::{Dims, LN_EPS};

/// An example: the tokens read and, for each position, the token its output is trained to give.
#[derive(Clone, Debug)]
pub struct Example {
    /// `ctx` tokens.
    pub tokens: Vec<usize>,
    /// `ctx` targets; only those at the supervised positions are read.
    pub targets: Vec<usize>,
}

/// What one block keeps of an example for its backward pass.
struct BlockTrace {
    x: Vec<f64>,
    q: Vec<f64>,
    kv: Vec<f64>,
    att: Vec<f64>,
    o: Vec<f64>,
    hat1: Vec<f64>,
    sigma1: Vec<f64>,
    y1: Vec<f64>,
    r: Vec<f64>,
    hat2: Vec<f64>,
    sigma2: Vec<f64>,
}

/// `y = x W` for `x` of `n` floats and `W`, `n` rows of `m`.
fn apply(x: &[f64], w: &[f64], m: usize) -> Vec<f64> {
    let mut y = vec![0.0; m];
    for (i, &xi) in x.iter().enumerate() {
        for o in 0..m {
            y[o] += xi * w[i * m + o];
        }
    }
    y
}

/// Layer normalization of `z` with gain `g` and bias `b`: the normalized vector, the standard
/// deviation with its floor, and the output.
fn normalize(z: &[f64], g: &[f64], b: &[f64]) -> (Vec<f64>, f64, Vec<f64>) {
    let n = z.len() as f64;
    let mean = z.iter().sum::<f64>() / n;
    let var = z.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    let sigma = (var + LN_EPS).sqrt();
    let hat: Vec<f64> = z.iter().map(|v| (v - mean) / sigma).collect();
    let y = hat.iter().zip(g).zip(b).map(|((h, g), b)| h * g + b).collect();
    (hat, sigma, y)
}

/// The forward pass of one example: the traces of the blocks and the last block's output,
/// `ctx` rows of `d`.
fn forward(dims: Dims, p: &[f64], tokens: &[usize]) -> (Vec<BlockTrace>, Vec<f64>) {
    let (d, t_len, dk, dff) = (dims.d, dims.ctx, dims.dk(), dims.dff);
    let pe = dims.positions();
    let mut x = vec![0.0; t_len * d];
    for t in 0..t_len {
        for i in 0..d {
            x[t * d + i] = p[dims.emb() + tokens[t] * d + i] + pe[t * d + i] as f64;
        }
    }
    let mut traces = Vec::new();
    for l in 0..dims.layers {
        let at = dims.block(l);
        let (mut q, mut kv) = (vec![0.0; t_len * d], vec![0.0; t_len * 2 * d]);
        for t in 0..t_len {
            let row = &x[t * d..(t + 1) * d];
            q[t * d..(t + 1) * d].copy_from_slice(&apply(row, &p[at.wq..at.wq + d * d], d));
            kv[t * 2 * d..(t + 1) * 2 * d].copy_from_slice(&apply(row, &p[at.wkv..at.wkv + d * 2 * d], 2 * d));
        }
        // att[t][h][j] for j <= t, zero past t: the causal mask.
        let mut att = vec![0.0; t_len * dims.heads * t_len];
        let mut o = vec![0.0; t_len * d];
        for t in 0..t_len {
            for h in 0..dims.heads {
                let mut scores = vec![0.0; t + 1];
                for (j, score) in scores.iter_mut().enumerate() {
                    for r in 0..dk {
                        *score += q[t * d + h * dk + r] * kv[j * 2 * d + h * dk + r];
                    }
                    *score /= (dk as f64).sqrt();
                }
                let top = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let total: f64 = scores.iter().map(|s| (s - top).exp()).sum();
                for j in 0..=t {
                    let a = (scores[j] - top).exp() / total;
                    att[(t * dims.heads + h) * t_len + j] = a;
                    for r in 0..dk {
                        o[t * d + h * dk + r] += a * kv[j * 2 * d + d + h * dk + r];
                    }
                }
            }
        }
        let (mut hat1, mut sigma1, mut y1) = (vec![0.0; t_len * d], vec![0.0; t_len], vec![0.0; t_len * d]);
        let (mut r_all, mut hat2, mut sigma2) = (vec![0.0; t_len * dff], vec![0.0; t_len * d], vec![0.0; t_len]);
        let mut next = vec![0.0; t_len * d];
        for t in 0..t_len {
            let ao = apply(&o[t * d..(t + 1) * d], &p[at.wo..at.wo + d * d], d);
            let z1: Vec<f64> = (0..d).map(|i| x[t * d + i] + ao[i]).collect();
            let (hat, sigma, y) = normalize(&z1, &p[at.g1..at.g1 + d], &p[at.b1..at.b1 + d]);
            let f1 = apply(&y, &p[at.w1..at.w1 + d * dff], dff);
            let r: Vec<f64> = f1.iter().map(|v| v.max(0.0)).collect();
            let f2 = apply(&r, &p[at.w2..at.w2 + dff * d], d);
            let z2: Vec<f64> = (0..d).map(|i| y[i] + f2[i]).collect();
            let (h2, s2, out) = normalize(&z2, &p[at.g2..at.g2 + d], &p[at.b2..at.b2 + d]);
            hat1[t * d..(t + 1) * d].copy_from_slice(&hat);
            sigma1[t] = sigma;
            y1[t * d..(t + 1) * d].copy_from_slice(&y);
            r_all[t * dff..(t + 1) * dff].copy_from_slice(&r);
            hat2[t * d..(t + 1) * d].copy_from_slice(&h2);
            sigma2[t] = s2;
            next[t * d..(t + 1) * d].copy_from_slice(&out);
        }
        traces.push(BlockTrace { x, q, kv, att, o, hat1, sigma1, y1, r: r_all, hat2, sigma2 });
        x = next;
    }
    (traces, x)
}

/// The logits of one example at every position: `ctx` rows of `vocab`.
pub fn logits(dims: Dims, p: &[f64], tokens: &[usize]) -> Vec<f64> {
    let (_, x) = forward(dims, p, tokens);
    let w = &p[dims.wout()..dims.wout() + dims.d * dims.vocab];
    (0..dims.ctx).flat_map(|t| apply(&x[t * dims.d..(t + 1) * dims.d], w, dims.vocab)).collect()
}

/// The cross-entropy of `row` against `target`, and the probabilities.
pub fn cross_entropy(row: &[f64], target: usize) -> (f64, Vec<f64>) {
    let top = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let total: f64 = row.iter().map(|v| (v - top).exp()).sum();
    let probabilities = row.iter().map(|v| (v - top).exp() / total).collect();
    (total.ln() + top - row[target], probabilities)
}

#[cfg(test)]
/// The loss of `F.cross_entropy(..., reduction="mean")` over the examples of `batch` and the
/// positions of `sup`.
pub fn loss(dims: Dims, p: &[f64], batch: &[Example], sup: &[usize]) -> f64 {
    let mut total = 0.0;
    for example in batch {
        let all = logits(dims, p, &example.tokens);
        for &pos in sup {
            total += cross_entropy(&all[pos * dims.vocab..(pos + 1) * dims.vocab], example.targets[pos]).0;
        }
    }
    total / (batch.len() * sup.len()) as f64
}

/// The gradient through a layer normalization: from the gradient `dy` of its output to that of
/// its input, adding the gain's and the bias's to `gg` and `gb`.
fn normalize_back(dy: &[f64], hat: &[f64], sigma: f64, g: &[f64], gg: &mut [f64], gb: &mut [f64]) -> Vec<f64> {
    let n = dy.len() as f64;
    let dhat: Vec<f64> = dy.iter().zip(g).map(|(a, b)| a * b).collect();
    for i in 0..dy.len() {
        gg[i] += dy[i] * hat[i];
        gb[i] += dy[i];
    }
    let mean = dhat.iter().sum::<f64>() / n;
    let along = dhat.iter().zip(hat).map(|(a, b)| a * b).sum::<f64>() / n;
    dhat.iter().zip(hat).map(|(a, h)| (a - mean - h * along) / sigma).collect()
}

/// Adds `x^T dy` to the gradient of `W`, `n` rows of `m`, and returns `dy W^T`.
fn apply_back(x: &[f64], dy: &[f64], w: &[f64], gw: &mut [f64]) -> Vec<f64> {
    let m = dy.len();
    let mut dx = vec![0.0; x.len()];
    for i in 0..x.len() {
        for o in 0..m {
            gw[i * m + o] += x[i] * dy[o];
            dx[i] += dy[o] * w[i * m + o];
        }
    }
    dx
}

/// The loss of [`loss`] and its gradient in every parameter.
pub fn loss_and_gradient(dims: Dims, p: &[f64], batch: &[Example], sup: &[usize]) -> (f64, Vec<f64>) {
    let (d, t_len, dk, dff, v) = (dims.d, dims.ctx, dims.dk(), dims.dff, dims.vocab);
    let scale = 1.0 / (batch.len() * sup.len()) as f64;
    let mut grad = vec![0.0; dims.len()];
    let mut total = 0.0;
    for example in batch {
        let (traces, last) = forward(dims, p, &example.tokens);
        let mut dx = vec![0.0; t_len * d];
        for &pos in sup {
            let row = apply(&last[pos * d..(pos + 1) * d], &p[dims.wout()..dims.wout() + d * v], v);
            let (ce, mut dlogits) = cross_entropy(&row, example.targets[pos]);
            total += ce;
            dlogits[example.targets[pos]] -= 1.0;
            dlogits.iter_mut().for_each(|g| *g *= scale);
            let (w, gw) = (&p[dims.wout()..dims.wout() + d * v], &mut grad[dims.wout()..dims.wout() + d * v]);
            let back = apply_back(&last[pos * d..(pos + 1) * d], &dlogits, w, gw);
            dx[pos * d..(pos + 1) * d].copy_from_slice(&back);
        }
        for l in (0..dims.layers).rev() {
            let (at, tr) = (dims.block(l), &traces[l]);
            let mut dprev = vec![0.0; t_len * d];
            let mut dq = vec![0.0; t_len * d];
            let mut dkv = vec![0.0; t_len * 2 * d];
            for t in 0..t_len {
                let dy2 = &dx[t * d..(t + 1) * d];
                let (g, b) = grad.split_at_mut(at.b2);
                let dz2 = normalize_back(
                    dy2,
                    &tr.hat2[t * d..(t + 1) * d],
                    tr.sigma2[t],
                    &p[at.g2..at.g2 + d],
                    &mut g[at.g2..at.g2 + d],
                    &mut b[..d],
                );
                let r = &tr.r[t * dff..(t + 1) * dff];
                let dr = apply_back(r, &dz2, &p[at.w2..at.w2 + dff * d], &mut grad[at.w2..at.w2 + dff * d]);
                let df1: Vec<f64> = dr.iter().zip(r).map(|(g, r)| if *r > 0.0 { *g } else { 0.0 }).collect();
                let y1 = &tr.y1[t * d..(t + 1) * d];
                let from_ffn = apply_back(y1, &df1, &p[at.w1..at.w1 + d * dff], &mut grad[at.w1..at.w1 + d * dff]);
                let dy1: Vec<f64> = (0..d).map(|i| dz2[i] + from_ffn[i]).collect();
                let (g, b) = grad.split_at_mut(at.b1);
                let dz1 = normalize_back(
                    &dy1,
                    &tr.hat1[t * d..(t + 1) * d],
                    tr.sigma1[t],
                    &p[at.g1..at.g1 + d],
                    &mut g[at.g1..at.g1 + d],
                    &mut b[..d],
                );
                let o = &tr.o[t * d..(t + 1) * d];
                let do_ = apply_back(o, &dz1, &p[at.wo..at.wo + d * d], &mut grad[at.wo..at.wo + d * d]);
                for i in 0..d {
                    dprev[t * d + i] += dz1[i];
                }
                for h in 0..dims.heads {
                    let a = &tr.att[(t * dims.heads + h) * t_len..(t * dims.heads + h + 1) * t_len];
                    let mut da = vec![0.0; t + 1];
                    for j in 0..=t {
                        for r in 0..dk {
                            da[j] += do_[h * dk + r] * tr.kv[j * 2 * d + d + h * dk + r];
                            dkv[j * 2 * d + d + h * dk + r] += a[j] * do_[h * dk + r];
                        }
                    }
                    let mean: f64 = (0..=t).map(|j| a[j] * da[j]).sum();
                    for j in 0..=t {
                        let ds = a[j] * (da[j] - mean) / (dk as f64).sqrt();
                        for r in 0..dk {
                            dq[t * d + h * dk + r] += ds * tr.kv[j * 2 * d + h * dk + r];
                            dkv[j * 2 * d + h * dk + r] += ds * tr.q[t * d + h * dk + r];
                        }
                    }
                }
            }
            for t in 0..t_len {
                let x = &tr.x[t * d..(t + 1) * d];
                let from_q =
                    apply_back(x, &dq[t * d..(t + 1) * d], &p[at.wq..at.wq + d * d], &mut grad[at.wq..at.wq + d * d]);
                let from_kv = apply_back(
                    x,
                    &dkv[t * 2 * d..(t + 1) * 2 * d],
                    &p[at.wkv..at.wkv + d * 2 * d],
                    &mut grad[at.wkv..at.wkv + d * 2 * d],
                );
                for i in 0..d {
                    dprev[t * d + i] += from_q[i] + from_kv[i];
                }
            }
            dx = dprev;
        }
        for t in 0..t_len {
            for i in 0..d {
                grad[dims.emb() + example.tokens[t] * d + i] += dx[t * d + i];
            }
        }
    }
    (total * scale, grad)
}

/// What a diagnosis reads of one example at one position, block by block. Only the
/// `diagnose` binary uses it.
#[allow(dead_code)]
pub struct Probe {
    /// For each block, which of its `dff` hidden units fire.
    pub active: Vec<Vec<bool>>,
    /// For each block, the attention weights of each head: `heads` rows of `pos + 1`.
    pub attention: Vec<Vec<f64>>,
    /// For each block, the floored standard deviations that its two normalizations divide by.
    pub sigma: Vec<(f64, f64)>,
}

/// The probe of one example at the position `pos`.
#[allow(dead_code)]
pub fn probe(dims: Dims, p: &[f64], tokens: &[usize], pos: usize) -> Probe {
    let (traces, _) = forward(dims, p, tokens);
    let mut out = Probe { active: Vec::new(), attention: Vec::new(), sigma: Vec::new() };
    for trace in traces.iter() {
        out.active.push(trace.r[pos * dims.dff..(pos + 1) * dims.dff].iter().map(|&v| v > 0.0).collect());
        let mut rows = Vec::new();
        for h in 0..dims.heads {
            let at = (pos * dims.heads + h) * dims.ctx;
            rows.extend_from_slice(&trace.att[at..at + pos + 1]);
        }
        out.attention.push(rows);
        out.sigma.push((trace.sigma1[pos], trace.sigma2[pos]));
    }
    out
}

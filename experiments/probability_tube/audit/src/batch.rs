//! Batch evaluation of EM1-EM4; only intermediate norms enter the radius formula.
//! Equations ported from probability_tube/encoder.rs at 7f6660d1.
use crate::algebra::linear;
use probability_tube::encoder::{Caps, EPSILON, Shape, norm};
const W: usize = 128;
const H: usize = 682;
pub struct Trace {
    pub norms: Vec<[f64; 9]>,
    pub features: Vec<f64>,
    pub logits: Vec<f64>,
    pub matrices: [f64; 3],
}
fn save_norms(x: &[f64], width: usize, norms: &mut [[f64; 9]], at: usize) {
    for (r, x) in x.chunks_exact(width).enumerate() {
        norms[r][at] = norm(x);
    }
}
fn rms(x: &mut [f64], width: usize) {
    for x in x.chunks_exact_mut(width) {
        let d = (x.iter().map(|v| v * v).sum::<f64>() / width as f64 + EPSILON).sqrt();
        for v in x {
            *v /= d;
        }
    }
}
pub fn forward(p: &[f64], rows: &[u32]) -> Trace {
    let shape = Shape::ORIGINAL;
    assert_eq!(p.len(), shape.count());
    let n = rows.len() / 3;
    let mut norms = vec![[0.; 9]; n];
    let mut input = Vec::with_capacity(n * W);
    for row in rows.as_chunks::<3>().0 {
        for k in 0..W {
            input.push(
                p[row[0] as usize * W + k]
                    + if k % 2 == 0 {
                        -p[row[1] as usize * W + k + 1]
                    } else {
                        p[row[1] as usize * W + k - 1]
                    },
            );
        }
    }
    save_norms(&input, W, &mut norms, 0);
    rms(&mut input, W);
    save_norms(&input, W, &mut norms, 1);
    let mut inner = input.clone();
    rms(&mut inner, W);
    save_norms(&inner, W, &mut norms, 2);
    let g = linear(&inner, n, W, &p[shape.gate()], H);
    let u = linear(&inner, n, W, &p[shape.value()], H);
    save_norms(&g, H, &mut norms, 3);
    save_norms(&u, H, &mut norms, 4);
    let mut hidden = vec![0.; n * H];
    for r in 0..n {
        for k in 0..H / 2 {
            let a = r * H + k;
            let b = a + H / 2;
            hidden[a] = (g[a] * u[a] - g[b] * u[b]) / 2f64.sqrt();
            hidden[b] = (g[a] * u[b] + g[b] * u[a]) / 2f64.sqrt();
        }
    }
    save_norms(&hidden, H, &mut norms, 5);
    let mut branch = linear(&hidden, n, H, &p[shape.output()], W);
    save_norms(&branch, W, &mut norms, 6);
    rms(&mut branch, W);
    let mut features: Vec<_> = input.iter().zip(branch).map(|(x, b)| x + b).collect();
    save_norms(&features, W, &mut norms, 7);
    rms(&mut features, W);
    save_norms(&features, W, &mut norms, 8);
    let logits = linear(&features, n, W, &p[..shape.embeddings()], 97);
    Trace {
        norms,
        features,
        logits,
        matrices: [
            norm(&p[shape.gate()]),
            norm(&p[shape.value()]),
            norm(&p[shape.output()]),
        ],
    }
}
fn rms_radius(norm: f64, width: usize, radius: f64) -> f64 {
    let width = width as f64;
    let s = (norm * norm + width * EPSILON).sqrt();
    let mut bound = (2. * width.sqrt()).min(2. * width.sqrt() * radius / s);
    if radius < s {
        bound = bound.min(width.sqrt() * radius / (s * (s - radius)).sqrt());
    }
    bound
}
impl Trace {
    pub fn feature_radii(&self, rows: &[u32], caps: &Caps) -> Vec<f64> {
        self.norms
            .iter()
            .zip(rows.as_chunks::<3>().0)
            .map(|(n, row)| {
                let r0 = caps.rows[row[0] as usize] + caps.rows[row[1] as usize];
                let r1 = rms_radius(n[0], W, r0);
                let r2 = rms_radius(n[1], W, r1);
                let rg = (self.matrices[0] + caps.gate) * r2 + caps.gate * n[2];
                let ru = (self.matrices[1] + caps.value) * r2 + caps.value * n[2];
                let rh = (rg * n[4] + n[3] * ru + rg * ru) / 2f64.sqrt();
                let rb = (self.matrices[2] + caps.output) * rh + caps.output * n[5];
                let rn = rms_radius(n[6], W, rb);
                rms_radius(n[7], W, r1 + rn)
            })
            .collect()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cached_batch_matches_scalar_bound_and_independent_forward() {
        crate::algebra::threads();
        let p: Vec<_> = crate::domain::model::initialize(crate::domain::model::Arm::Plain, 91)
            .into_iter()
            .map(|v| v as f64)
            .collect();
        let rows = [0, 0, 0, 1, 3, 0, 91, 17, 0];
        let trace = forward(&p, &rows);
        let oracle = crate::oracle::logits(crate::domain::model::Arm::Plain, &p, &rows);
        for (a, b) in trace.logits.iter().zip(oracle) {
            assert!((a - b).abs() < 1e-11);
        }
        for scale in [0., 1e-8, 1e-4, 0.1] {
            let caps = Caps {
                rows: (0..97).map(|c| scale * (1. + c as f64 / 97.)).collect(),
                gate: scale,
                value: 2. * scale,
                output: 3. * scale,
            };
            let radii = trace.feature_radii(&rows, &caps);
            for (i, row) in rows.as_chunks::<3>().0.iter().enumerate() {
                let f = probability_tube::encoder::forecast(
                    Shape::ORIGINAL,
                    &p,
                    row[0] as usize,
                    row[1] as usize,
                    &caps,
                );
                assert!((f.feature().radius - radii[i]).abs() < 1e-11 * (1. + radii[i]));
                for (a, b) in f
                    .feature()
                    .old
                    .iter()
                    .zip(&trace.features[i * W..(i + 1) * W])
                {
                    assert!((a - b).abs() < 1e-11);
                }
            }
        }
    }
}

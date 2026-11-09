//! FP64 old-node statistics for PROTOCOL.md; only operand pairs are accepted.
//! Source equations match flat_approx/fixtures/reference.rs at 60483223;
//! batched BLAS replaces scalar dense loops, without tangent products.
use crate::{
    algebra::linear,
    domain::model::{Arm, E, W},
    forecast_jet::{EPS, norm},
};
#[derive(Clone, Copy, Default)]
pub struct Node {
    pub denominators: [f64; 4],
    pub first_input: f64,
    pub gate_max: f64,
    pub gated_max: f64,
    pub value_max: f64,
    pub hidden_norm: f64,
}
pub struct Graph {
    pub nodes: Vec<Node>,
    pub features: Vec<f64>,
}
fn rms(x: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut y = Vec::with_capacity(x.len());
    let mut s = Vec::with_capacity(x.len() / W);
    for row in x.chunks_exact(W) {
        let scale = (row.iter().map(|v| v * v).sum::<f64>() / W as f64 + EPS).sqrt();
        s.push(scale);
        y.extend(row.iter().map(|v| v / scale));
    }
    (y, s)
}
fn maximum(x: &[f64], complex: bool) -> f64 {
    if complex {
        let h = x.len() / 2;
        (0..h).map(|i| x[i].hypot(x[i + h])).fold(0., f64::max)
    } else {
        x.iter().map(|x| x.abs()).fold(0., f64::max)
    }
}
pub fn graph(arm: Arm, p: &[f64], pairs: &[[u32; 2]]) -> Graph {
    let n = pairs.len();
    let h = arm.hidden();
    let mut nodes = vec![Node::default(); n];
    let embed: Vec<_> = pairs
        .iter()
        .flat_map(|r| {
            (0..W).map(move |k| {
                p[r[0] as usize * W + k]
                    + if k % 2 == 0 {
                        -p[r[1] as usize * W + k + 1]
                    } else {
                        p[r[1] as usize * W + k - 1]
                    }
            })
        })
        .collect();
    let (x0, s0) = rms(&embed);
    let (n1, s1) = rms(&x0);
    let off = E + 2 * h * W;
    let gv = linear(&n1, n, W, &p[E..off], 2 * h);
    let mut hidden = vec![0.; n * h];
    for q in 0..n {
        let a = &gv[q * 2 * h..q * 2 * h + h];
        let b = &gv[q * 2 * h + h..(q + 1) * 2 * h];
        let gated: Vec<_> = if arm == Arm::Swiglu {
            a.iter().map(|x| x / (1. + (-x).exp())).collect()
        } else if arm == Arm::Plain {
            a.to_vec()
        } else {
            (0..h)
                .map(|k| {
                    let at = k % (h / 2);
                    let length = (a[at].powi(2) + a[at + h / 2].powi(2) + EPS).sqrt();
                    a[k] / (1. + (-length).exp())
                })
                .collect()
        };
        for k in 0..h {
            hidden[q * h + k] = match arm {
                Arm::Swiglu | Arm::Diagonal => gated[k] * b[k],
                _ => {
                    if k < h / 2 {
                        (gated[k] * b[k] - gated[k + h / 2] * b[k + h / 2]) / 2f64.sqrt()
                    } else {
                        (gated[k - h / 2] * b[k] + gated[k] * b[k - h / 2]) / 2f64.sqrt()
                    }
                }
            };
        }
        nodes[q] = Node {
            denominators: [s0[q], s1[q], 0., 0.],
            first_input: norm(&n1[q * W..(q + 1) * W]),
            gate_max: maximum(a, arm != Arm::Swiglu),
            gated_max: maximum(&gated, arm != Arm::Swiglu),
            value_max: maximum(b, arm != Arm::Swiglu),
            hidden_norm: norm(&hidden[q * h..(q + 1) * h]),
        };
    }
    let raw = linear(&hidden, n, h, &p[off..], W);
    let (branch, s2) = rms(&raw);
    let residual: Vec<_> = x0.iter().zip(&branch).map(|(a, b)| a + b).collect();
    let (features, s3) = rms(&residual);
    for q in 0..n {
        nodes[q].denominators[2] = s2[q];
        nodes[q].denominators[3] = s3[q];
    }
    Graph { nodes, features }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{domain::model, oracle};
    #[test]
    fn old_graph_matches_independent_scalar_source_for_all_arms() {
        for arm in Arm::ALL {
            let p: Vec<_> = model::initialize(arm, 7)
                .iter()
                .map(|&x| x as f64)
                .collect();
            let pairs = [[0, 1], [9, 96], [52, 4]];
            let f = graph(arm, &p, &pairs);
            let actual = linear(&f.features, 3, W, &p[..E], 97);
            let expected = oracle::logits(arm, &p, &[0, 1, 0, 9, 96, 0, 52, 4, 0]);
            assert!(
                actual
                    .iter()
                    .zip(expected)
                    .all(|(a, b)| (a - b).abs() < 1e-11)
            );
        }
    }
}

//! Independent FP64 objective evaluations for finite differences. Source graph:
//! flat_approx/fixtures/reference.rs (9328bc83); deviations are the two fixed
//! auxiliary heads and boundary-specific objectives declared in PROTOCOL.md.
use crate::{
    algebra::linear,
    config::Prototypes,
    domain::model::{Arm, E, W},
    loss,
};
pub const EPS: f64 = f32::EPSILON as f64;
pub fn rms(x: &[f64], width: usize) -> Vec<f64> {
    x.chunks_exact(width)
        .flat_map(|row| {
            let s = (row.iter().map(|x| x * x).sum::<f64>() / width as f64 + EPS).sqrt();
            row.iter().map(move |x| x / s)
        })
        .collect()
}
pub fn embedding(p: &[f64], rows: &[u32]) -> Vec<f64> {
    rows.as_chunks::<3>()
        .0
        .iter()
        .flat_map(|row| {
            (0..W).map(move |k| {
                p[row[0] as usize * W + k]
                    + if k % 2 == 0 {
                        -p[row[1] as usize * W + k + 1]
                    } else {
                        p[row[1] as usize * W + k - 1]
                    }
            })
        })
        .collect()
}
pub fn hidden(arm: Arm, p: &[f64], n1: &[f64], batch: usize) -> Vec<f64> {
    let h = arm.hidden();
    let gv = linear(n1, batch, W, &p[E..E + 2 * h * W], 2 * h);
    let mut out = vec![0.; batch * h];
    for r in 0..batch {
        let x = &gv[r * 2 * h..(r + 1) * 2 * h];
        if arm == Arm::Swiglu {
            for k in 0..h {
                out[r * h + k] = x[k] * loss::sigmoid(x[k]) * x[k + h];
            }
        } else {
            for k in 0..h / 2 {
                let [a, b, u, v] = [x[k], x[k + h / 2], x[k + h], x[k + 3 * h / 2]];
                let gate = if arm == Arm::Plain {
                    1.
                } else {
                    loss::sigmoid((a * a + b * b + EPS).sqrt())
                };
                if arm == Arm::Diagonal {
                    out[r * h + k] = gate * a * u;
                    out[r * h + k + h / 2] = gate * b * v;
                } else {
                    out[r * h + k] = gate * (a * u - b * v) / 2f64.sqrt();
                    out[r * h + k + h / 2] = gate * (a * v + b * u) / 2f64.sqrt();
                }
            }
        }
    }
    out
}
pub fn final_features(arm: Arm, p: &[f64], x0: &[f64], hidden: &[f64], batch: usize) -> Vec<f64> {
    let branch = rms(
        &linear(
            hidden,
            batch,
            arm.hidden(),
            &p[E + 2 * arm.hidden() * W..],
            W,
        ),
        W,
    );
    let residual: Vec<_> = branch.iter().zip(x0).map(|(a, b)| a + b).collect();
    rms(&residual, W)
}
pub struct Cache {
    pub x0: Vec<f64>,
    pub n1: Vec<f64>,
    pub hidden: Vec<f64>,
    pub last: Vec<f64>,
}
impl Cache {
    pub fn new(arm: Arm, p: &[f64], rows: &[u32]) -> Self {
        let batch = rows.len() / 3;
        let x0 = rms(&embedding(p, rows), W);
        let n1 = rms(&x0, W);
        let hidden = hidden(arm, p, &n1, batch);
        let last = final_features(arm, p, &x0, &hidden, batch);
        Self {
            x0,
            n1,
            hidden,
            last,
        }
    }
}
pub fn pair_mean(scores: &[f64], labels: &[usize]) -> f64 {
    scores
        .as_chunks::<97>()
        .0
        .iter()
        .zip(labels)
        .map(|(z, &y)| loss::pairwise(z, y, 1.).0)
        .sum::<f64>()
        / labels.len() as f64
}
pub struct Heads {
    pub r0: Vec<f64>,
    pub r1: Vec<f64>,
}
impl From<&Prototypes> for Heads {
    fn from(p: &Prototypes) -> Self {
        Self {
            r0: p.vocabulary.iter().map(|&x| x as f64).collect(),
            r1: p.hidden.iter().map(|&x| x as f64).collect(),
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub enum Objective {
    Vocabulary,
    Readout,
    A,
    B,
    SharedE,
    Pair,
    Auxiliary,
}
pub fn objective(
    kind: Objective,
    arm: Arm,
    p: &[f64],
    rows: &[u32],
    old: &Cache,
    head: &Heads,
) -> f64 {
    let batch = rows.len() / 3;
    let labels: Vec<_> = rows
        .as_chunks::<3>()
        .0
        .iter()
        .map(|r| r[2] as usize)
        .collect();
    let vocabulary = || {
        pair_mean(
            &linear(&rms(&p[..E], W), 97, W, &head.r0, 97),
            &(0..97).collect::<Vec<_>>(),
        )
    };
    let readout = |h: &[f64]| pair_mean(&linear(h, batch, W, &p[..E], 97), &labels);
    let auxiliary = |h: &[f64]| {
        pair_mean(
            &linear(&rms(h, arm.hidden()), batch, arm.hidden(), &head.r1, 97),
            &labels,
        )
    };
    match kind {
        Objective::Vocabulary => vocabulary(),
        Objective::Readout => readout(&old.last),
        Objective::A => auxiliary(&hidden(arm, p, &old.n1, batch)),
        Objective::B => readout(&final_features(arm, p, &old.x0, &old.hidden, batch)),
        Objective::SharedE => vocabulary() + readout(&old.last),
        Objective::Pair => readout(&Cache::new(arm, p, rows).last),
        Objective::Auxiliary => {
            let current = Cache::new(arm, p, rows);
            vocabulary() + auxiliary(&current.hidden) + readout(&current.last)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn full_fp64_graph_matches_independent_scalar_source_for_every_arm() {
        let rows = [0, 0, 0, 1, 96, 0, 35, 48, 42];
        for arm in Arm::ALL {
            let p: Vec<_> = crate::domain::model::initialize(arm, 7)
                .iter()
                .map(|&x| x as f64)
                .collect();
            let h = Cache::new(arm, &p, &rows);
            let actual = linear(&h.last, 3, W, &p[..E], 97);
            let expected = crate::oracle::logits(arm, &p, &rows);
            assert!(
                actual
                    .iter()
                    .zip(expected)
                    .all(|(&a, b)| (a - b).abs() < 2e-12)
            );
        }
    }
}

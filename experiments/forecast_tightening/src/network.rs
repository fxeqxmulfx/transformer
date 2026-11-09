//! Source graph and task #188 tangents, with four radius tracks and saved stages.
use crate::{
    domain::model::{Arm, E, W},
    forecast_jet::{Jet, Product},
    jet::Tight,
    matrix::{self, Bound},
    product::product,
};
pub struct Factors {
    pub first: Bound,
    pub second: Bound,
}
impl Factors {
    pub fn new(arm: Arm, next: &[f64]) -> Self {
        let h = arm.hidden();
        let off = E + 2 * h * W;
        Self {
            first: matrix::bound(&next[E..off], 2 * h, W),
            second: matrix::bound(&next[off..], W, h),
        }
    }
}
pub struct Stage {
    pub name: &'static str,
    pub width: usize,
    pub center: Vec<f64>,
    pub radii: [Vec<f64>; 4],
    pub defects: Vec<f64>,
    pub gains: [Vec<f64>; 4],
}
impl Stage {
    fn capture(name: &'static str, t: &Tight) -> Self {
        Self {
            name,
            width: t.base.width,
            center: t.base.a.iter().zip(&t.base.v).map(|(a, v)| a + v).collect(),
            radii: t.radii.clone(),
            defects: t.defects.clone(),
            gains: t.gains.clone(),
        }
    }
}
pub struct Network {
    pub output: Tight,
    pub pre: Tight,
    pub stages: Vec<Stage>,
}
pub fn features(arm: Arm, old: &[f64], step: &[f64], rows: &[u32], factors: &Factors) -> Network {
    let mut stages = vec![];
    let gather = |p: &[f64]| {
        rows.as_chunks::<3>()
            .0
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
            .collect()
    };
    let input = Tight::initial(Jet {
        a: gather(old),
        v: gather(step),
        r: vec![0.; rows.len() / 3],
        width: W,
    });
    stages.push(Stage::capture("gather", &input));
    let input = input.rms();
    stages.push(Stage::capture("embedding-rms", &input));
    let normalized = input.rms();
    stages.push(Stage::capture("pre-rms", &normalized));
    let h = arm.hidden();
    let off = E + 2 * h * W;
    let gv = normalized.dense(&old[E..off], &step[E..off], 2 * h, factors.first);
    stages.push(Stage::capture("first-linear", &gv));
    let a = gv.split(0, h);
    let b = gv.split(h, h);
    let hidden = if arm == Arm::Swiglu {
        let gate = a.sigmoid();
        stages.push(Stage::capture("sigmoid", &gate));
        let gated = product(Product::Real, &a, &gate);
        stages.push(Stage::capture("gated", &gated));
        product(Product::Real, &gated, &b)
    } else {
        let gated = if arm == Arm::Plain {
            a
        } else {
            let radial = a.radial();
            stages.push(Stage::capture("radial", &radial));
            let gate = radial.sigmoid();
            stages.push(Stage::capture("sigmoid", &gate));
            product(Product::Gate, &a, &gate)
        };
        stages.push(Stage::capture("gated", &gated));
        product(
            if arm == Arm::Diagonal {
                Product::Diagonal
            } else {
                Product::Complex
            },
            &gated,
            &b,
        )
    };
    stages.push(Stage::capture("product", &hidden));
    let branch = hidden.dense(&old[off..], &step[off..], W, factors.second);
    stages.push(Stage::capture("second-linear", &branch));
    let branch = branch.rms();
    stages.push(Stage::capture("branch-rms", &branch));
    let pre = input.add(&branch);
    stages.push(Stage::capture("residual", &pre));
    let output = pre.rms();
    stages.push(Stage::capture("final-rms", &output));
    Network {
        output,
        pre,
        stages,
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{domain::model, forecast_jet::norm, forecast_network};
    #[test]
    fn all_arms_keep_the_old_tangent_and_enclose_nonzero_source_steps() {
        for arm in Arm::ALL {
            let old: Vec<_> = model::initialize(arm, 7)
                .iter()
                .map(|&x| x as f64)
                .collect();
            let step: Vec<_> = (0..old.len())
                .map(|i| 1e-4 * ((i % 7) as f64 - 3.))
                .collect();
            let next: Vec<_> = old.iter().zip(&step).map(|(a, d)| a + d).collect();
            let rows = [0, 1, 1, 5, 9, 14];
            let factors = Factors::new(arm, &next);
            let f = features(arm, &old, &step, &rows, &factors);
            let source = forecast_network::features(arm, &old, &step, &rows);
            assert_eq!(f.output.base.a, source.a);
            assert_eq!(f.output.base.v, source.v);
            assert_eq!(f.output.radii[0], source.r);
            let actual = features(arm, &next, &vec![0.; next.len()], &rows, &factors);
            for (before, after) in f.stages.iter().zip(actual.stages) {
                for q in 0..2 {
                    let w = before.width;
                    let error = norm(
                        &(0..w)
                            .map(|k| after.center[q * w + k] - before.center[q * w + k])
                            .collect::<Vec<_>>(),
                    );
                    assert!(
                        before
                            .radii
                            .iter()
                            .all(|r| error <= r[q] + 1e-8 * (1. + r[q])),
                        "{}",
                        before.name
                    );
                }
            }
        }
    }
}

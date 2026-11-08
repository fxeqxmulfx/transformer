//! Old-network tangent and finite propagation error; no candidate-network evaluation.
use crate::{
    domain::{
        Result,
        model::{Arm, E, W},
    },
    forecast_jet::{Jet, Product},
    inputs::require,
    oracle,
};

pub fn features(arm: Arm, old: &[f64], step: &[f64], rows: &[u32]) -> Jet {
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
    let input = Jet {
        a: gather(old),
        v: gather(step),
        r: vec![0.; rows.len() / 3],
        width: W,
    }
    .rms();
    let h = arm.hidden();
    let off = E + 2 * h * W;
    let gv = input.rms().dense(&old[E..off], &step[E..off], 2 * h);
    let a = gv.split(0, h);
    let b = gv.split(h, h);
    let hidden = if arm == Arm::Swiglu {
        Product::Real.jets(&Product::Real.jets(&a, &a.sigmoid()), &b)
    } else {
        let gated = if arm == Arm::Plain {
            a
        } else {
            Product::Gate.jets(&a, &a.radial().sigmoid())
        };
        if arm == Arm::Diagonal {
            Product::Diagonal
        } else {
            Product::Complex
        }
        .jets(&gated, &b)
    };
    let branch = hidden.dense(&old[off..], &step[off..], W).rms();
    input.add(&branch).rms()
}

pub fn tangent_check(arm: Arm, old: &[f64], step: &[f64], rows: &[u32]) -> Result<f64> {
    let jet = features(arm, old, step, rows);
    let h = 1e-3;
    let plus: Vec<_> = old.iter().zip(step).map(|(a, d)| a + h * d).collect();
    let minus: Vec<_> = old.iter().zip(step).map(|(a, d)| a - h * d).collect();
    let zp = oracle::logits(arm, &plus, rows);
    let zm = oracle::logits(arm, &minus, rows);
    let mut max = 0f64;
    for q in 0..jet.rows() {
        for c in 0..97 {
            let tangent = (0..W)
                .map(|k| old[c * W + k] * jet.v[q * W + k] + step[c * W + k] * jet.a[q * W + k])
                .sum::<f64>();
            let finite = (zp[q * 97 + c] - zm[q * 97 + c]) / (2. * h);
            let error = (finite - tangent).abs() / (1. + finite.abs().max(tangent.abs()));
            max = max.max(error);
        }
    }
    require(
        max < 2e-6,
        "forecast tangent disagrees with independent scalar central differences",
    )?;
    Ok(max)
}

pub fn canonical_rows(data: &crate::domain::data::Data) -> Vec<u32> {
    let mut rows = vec![0; 9409 * 3];
    for r in data
        .train
        .as_chunks::<3>()
        .0
        .iter()
        .chain(data.validation.as_chunks::<3>().0)
    {
        let q = (97 * r[0] + r[1]) as usize;
        rows[q * 3..q * 3 + 3].copy_from_slice(r);
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::model;
    use crate::forecast_jet::norm;
    #[test]
    fn all_source_arms_have_nonzero_correct_tangents_and_valid_balls() {
        for arm in Arm::ALL {
            let old: Vec<_> = model::initialize(arm, 7)
                .iter()
                .map(|&x| x as f64)
                .collect();
            let step: Vec<_> = (0..old.len())
                .map(|k| 0.0001 * ((k % 7) as f64 - 3.))
                .collect();
            let rows = [0, 1, 1, 5, 9, 14];
            tangent_check(arm, &old, &step, &rows).unwrap();
            let f = features(arm, &old, &step, &rows);
            let new: Vec<_> = old.iter().zip(&step).map(|(a, d)| a + d).collect();
            let actual = features(arm, &new, &vec![0.; new.len()], &rows);
            assert!(norm(&f.v) > 0.);
            for q in 0..2 {
                let error: Vec<_> = (0..W)
                    .map(|k| actual.a[q * W + k] - f.a[q * W + k] - f.v[q * W + k])
                    .collect();
                assert!(norm(&error) <= f.r[q] + 1e-10);
                assert!(f.r[q] > 0.);
            }
        }
    }
}

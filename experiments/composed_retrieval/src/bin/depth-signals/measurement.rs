//! Actual complete-vocabulary loss along the scalar depth line of CM114.
use baseline::channel::{argmax, cross_entropy};
use composed_retrieval::{
    Result,
    model::Model,
    operator::{Tables, softmax},
    train::Sample,
};
use serde::Serialize;

#[derive(Default, Serialize)]
pub struct Measurement {
    pub targets: usize,
    pub depth: f64,
    pub product: f64,
    pub slope: f64,
    pub feasible_secant: f64,
    pub ce: f64,
    /// In this order, use depth zero, one half and one with all other weights fixed.
    pub ce_at_depth: [f64; 3],
    pub correct_at_depth: [usize; 3],
    pub target_margin_at_depth: [f64; 3],
    pub mean_direction_range: f64,
    pub distinguishable_rows: usize,
}
fn range(row: &[f64]) -> f64 {
    row.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - row.iter().copied().fold(f64::INFINITY, f64::min)
}
pub fn measure(m: &Model, t: &Tables, rows: &[&Sample], eps: f64) -> Result<Measurement> {
    let depth = m.weights[t.layout.depth];
    let product = m.weights[t.layout.product];
    if rows.is_empty() || m.vocabulary < 2 || !(eps > 0. && eps.is_finite()) {
        return Err("need observed rows, rival class and a finite positive step".into());
    }
    let lo = (depth - eps).max(0.);
    let hi = (depth + eps).min(1.);
    let mut value = Measurement {
        targets: rows.len(),
        depth,
        product,
        ..Measurement::default()
    };
    let mut trial_ce = [0.; 5];
    for row in rows {
        let p = t.pass(m, &row.prefix);
        let z = t.logits(m, &p.hidden);
        let first = t.logits(m, &p.s1);
        let second = t.logits(m, &p.s2);
        let direction: Vec<_> = second
            .iter()
            .zip(&first)
            .map(|(b, a)| (1. - product) * (b - a))
            .collect();
        let spread = range(&direction);
        value.mean_direction_range += spread;
        value.distinguishable_rows += usize::from(spread > 1e-10);
        let mut probabilities = z.clone();
        softmax(&mut probabilities);
        value.slope += probabilities
            .iter()
            .zip(&direction)
            .map(|(p, d)| p * d)
            .sum::<f64>()
            - direction[row.target];
        value.ce += cross_entropy(&z, row.target);
        for (k, h) in [0., 0.5, 1., lo, hi].into_iter().enumerate() {
            let trial: Vec<_> = z
                .iter()
                .zip(&direction)
                .map(|(z, d)| z + (h - depth) * d)
                .collect();
            trial_ce[k] += cross_entropy(&trial, row.target);
            if k < 3 {
                value.correct_at_depth[k] += usize::from(argmax(&trial) == row.target);
                let rival = trial
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != row.target)
                    .map(|(_, v)| *v)
                    .fold(f64::NEG_INFINITY, f64::max);
                value.target_margin_at_depth[k] += trial[row.target] - rival;
            }
        }
    }
    let n = value.targets as f64;
    value.ce /= n;
    value.slope /= n;
    value.mean_direction_range /= n;
    for v in &mut trial_ce {
        *v /= n;
    }
    for margin in &mut value.target_margin_at_depth {
        *margin /= n;
    }
    value.ce_at_depth.copy_from_slice(&trial_ce[..3]);
    value.feasible_secant = (trial_ce[4] - trial_ce[3]) / (hi - lo);
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use composed_retrieval::data;
    #[test]
    fn gate_line_agrees_with_rebuilding_the_actual_model() -> Result<()> {
        let mut m = Model::new(5, 5, &data::protocol()?, 1)?;
        let l = m.validate()?;
        for i in l.relation.clone() {
            m.weights[i] = (i as f64).sin() * 3.;
        }
        let rows = [
            Sample {
                prefix: vec![],
                target: 1,
            },
            Sample {
                prefix: vec![1],
                target: 2,
            },
            Sample {
                prefix: vec![2, 1],
                target: 3,
            },
            Sample {
                prefix: vec![1, 2, 3, 1, 2],
                target: 3,
            },
        ];
        let rows: Vec<_> = rows.iter().collect();
        let lengths = [0, 1, 4];
        for base in [0., 0.43, 1.] {
            m.weights[l.depth] = base;
            let obs = measure(&m, &Tables::new(&m, &lengths)?, &rows, 0.0001)?;
            assert!(obs.mean_direction_range > 0.01);
            for (j, h) in [0., 0.5, 1.].into_iter().enumerate() {
                let mut other = m.clone();
                other.weights[l.depth] = h;
                let t = Tables::new(&other, &lengths)?;
                let ce = rows
                    .iter()
                    .map(|s| {
                        cross_entropy(
                            &t.logits(&other, &t.pass(&other, &s.prefix).hidden),
                            s.target,
                        )
                    })
                    .sum::<f64>()
                    / rows.len() as f64;
                assert!((obs.ce_at_depth[j] - ce).abs() < 1e-12);
                assert!(ce + 1e-12 >= obs.ce + (h - base) * obs.slope);
            }
        }
        Ok(())
    }
}

//! Independent FP64 graph evaluations, differentiated by central differences.
//! Stable CE/pair formulas share logits but never use the GPU derivative.
use crate::{
    algebra,
    config::Kernels,
    coordinates as coord,
    domain::{
        Result,
        model::{Arm, E, W},
    },
    infrastructure::engine::Engine,
    measure::{Cost, Measured},
    reference,
};
use serde_json::{Value, json};

pub fn values(arm: Arm, p: &[f64], rows: &[u32], cost: &mut Cost) -> [f64; 2] {
    cost.cpu_forward_calls += 1;
    cost.cpu_forward_rows += rows.len() / 3;
    let cache = reference::Cache::new(arm, p, rows);
    let logits = algebra::linear(&cache.last, rows.len() / 3, W, &p[..E], 97);
    let labels: Vec<_> = rows
        .as_chunks::<3>()
        .0
        .iter()
        .map(|r| r[2] as usize)
        .collect();
    let ce = logits
        .as_chunks::<97>()
        .0
        .iter()
        .zip(&labels)
        .map(|(z, &y)| {
            let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            max + z.iter().map(|x| (x - max).exp()).sum::<f64>().ln() - z[y]
        })
        .sum::<f64>()
        / labels.len() as f64;
    [ce, reference::pair_mean(&logits, &labels)]
}
fn checks(
    arm: Arm,
    p: &[f64],
    rows: &[u32],
    direction: &[f64],
    gradient: &[Vec<f64>; 2],
    cost: &mut Cost,
) -> Vec<Value> {
    let expanded = coord::expand(arm, direction);
    let plus: Vec<_> = p.iter().zip(&expanded).map(|(p, d)| p + 1e-4 * d).collect();
    let minus: Vec<_> = p.iter().zip(&expanded).map(|(p, d)| p - 1e-4 * d).collect();
    let a = values(arm, &plus, rows, cost);
    let b = values(arm, &minus, rows, cost);
    (0..2)
        .map(|j| {
            let finite = (a[j] - b[j]) / 2e-4;
            let analytic = coord::dot(&coord::pullback(arm, &gradient[j]), direction);
            let error = (finite - analytic).abs();
            let tolerance = 2e-5 + 2e-3 * finite.abs().max(analytic.abs());
            json!({"objective":(["ce","pair"][j]),"finite":finite,"analytic":analytic,
            "error":error,"tolerance":tolerance,"passed":error.is_finite()&&error<=tolerance})
        })
        .collect()
}
pub fn short(
    engine: &Engine,
    kernels: &Kernels,
    p: &[f64],
    data: &crate::domain::data::Data,
    seed: u64,
    cost: &mut Cost,
) -> Result<Vec<Value>> {
    let mut report = vec![];
    for (validation, rows) in [(false, &data.train[..24]), (true, &data.validation[..24])] {
        engine.forward(validation, 0, 8, false)?;
        cost.forward_batches += 1;
        engine.backward(8)?;
        cost.backward_batches += 1;
        let ce = engine.gradients.read()?.iter().map(|&v| v as f64).collect();
        engine.pair_scores(kernels, 8)?;
        engine.backward(8)?;
        cost.backward_batches += 1;
        let pair = engine.gradients.read()?.iter().map(|&v| v as f64).collect();
        let g = [ce, pair];
        for group in 0..3 {
            for repetition in 0..2 {
                let d = coord::direction(
                    group,
                    seed ^ 0x616c69676e000000 ^ ((group as u64) << 16) ^ repetition,
                );
                for mut row in checks(engine.arm, p, rows, &d, &g, cost) {
                    row["scope"] = json!(if validation {
                        "validation-first8"
                    } else {
                        "train-first8"
                    });
                    row["group"] = json!(coord::groups()[group].0);
                    row["direction"] = json!(repetition);
                    report.push(row);
                }
            }
        }
    }
    Ok(report)
}
pub fn full(
    arm: Arm,
    p: &[f64],
    rows: &[u32],
    delta: &[f64],
    measured: &Measured,
    cost: &mut Cost,
) -> Vec<Value> {
    let mut direction = coord::owners(arm, delta);
    let norm = coord::norm(&direction);
    if norm == 0. {
        return vec![json!({"zero_direction":true,"passed":true})];
    }
    for x in &mut direction {
        *x /= norm;
    }
    checks(arm, p, rows, &direction, &measured.gradients, cost)
}

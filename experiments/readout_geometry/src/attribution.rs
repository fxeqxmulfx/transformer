//! Exact signed component attribution of the known source step. The partial
//! readout/encoder/sampling/memory split is task #186's, with encoder feature
//! effects propagated for each component before adding the mixed readout term.
use crate::{
    algebra::linear,
    domain::{
        Result,
        model::{Arm, E, W},
    },
    forcing::Direction,
    forecast_network,
    inputs::require,
    network::Network,
};
use serde_json::{Value, json};
pub const NAMES: [&str; 8] = [
    "decay",
    "readout_credit",
    "encoder_credit",
    "sampling",
    "moment_memory",
    "rounding",
    "projection_parameters",
    "mixed_readout_feature",
];
pub struct Problem<'a> {
    pub prior: &'a [f64],
    pub next: &'a [f64],
    pub direction: &'a Direction,
    pub forecast: &'a Network,
    pub rows: &'a [u32],
    pub labels: &'a [Option<usize>],
    pub splits: &'a [Vec<usize>; 2],
}
pub fn prepare(p: Problem<'_>) -> Result<Value> {
    let Problem {
        prior,
        next,
        direction,
        forecast,
        rows,
        labels,
        splits,
    } = p;
    let n = rows.len() / 3;
    let mut steps = vec![vec![0.; prior.len()]; 7];
    for k in 0..E {
        steps[0][k] = -direction.eta * prior[k];
        for (part, step) in steps.iter_mut().enumerate().take(5).skip(1) {
            step[k] = -direction.eta * direction.diagonal[k] * direction.components[part - 1][k];
        }
        steps[5][k] = direction.rounding[k];
    }
    for k in E..prior.len() {
        steps[6][k] = next[k] - prior[k];
    }
    let parameter_error = (0..prior.len())
        .map(|k| (steps.iter().map(|s| s[k]).sum::<f64>() - (next[k] - prior[k])).abs())
        .fold(0., f64::max);
    require(
        parameter_error < 1e-10,
        "parameter component closure failed",
    )?;
    let old = linear(&forecast.output.base.a, n, W, &prior[..E], 97);
    let mut contributions = vec![];
    let mut tangent_sum = vec![0.; n * W];
    for step in &steps {
        let component = forecast_network::features(Arm::Plain, prior, step, rows);
        for (sum, v) in tangent_sum.iter_mut().zip(&component.v) {
            *sum += v;
        }
        let direct = linear(&forecast.output.base.a, n, W, &step[..E], 97);
        let feature = linear(&component.v, n, W, &prior[..E], 97);
        contributions.push(
            direct
                .iter()
                .zip(&feature)
                .map(|(a, b)| a + b)
                .collect::<Vec<_>>(),
        );
    }
    let tangent_error = tangent_sum
        .iter()
        .zip(&forecast.output.base.v)
        .map(|(a, b)| (a - b).abs() / (1. + a.abs().max(b.abs())))
        .fold(0., f64::max);
    require(
        tangent_error < 1e-8,
        "component tangents do not sum to actual tangent",
    )?;
    let delta: Vec<_> = next[..E].iter().zip(prior).map(|(a, b)| a - b).collect();
    contributions.push(linear(&forecast.output.base.v, n, W, &delta, 97));
    let center: Vec<_> = forecast
        .output
        .base
        .a
        .iter()
        .zip(&forecast.output.base.v)
        .map(|(a, v)| a + v)
        .collect();
    let predicted = linear(&center, n, W, &next[..E], 97);
    let closure = (0..n * 97)
        .map(|i| {
            let sum: f64 = contributions.iter().map(|v| v[i]).sum();
            (sum - (predicted[i] - old[i])).abs()
                / (1. + sum.abs() + predicted[i].abs() + old[i].abs())
        })
        .fold(0., f64::max);
    require(closure < 1e-8, "component logit changes do not close")?;
    let mut result = json!({"parameter_closure_max_error":parameter_error,"tangent_closure_scaled_error":tangent_error,
        "logit_closure_scaled_error":closure,"direction":direction.summary(),"names":NAMES,"frozen_before_updated_forward":true});
    let mut records = vec![];
    for (name, split) in ["train", "validation"].into_iter().zip(splits) {
        let mut sums = [0.; 8];
        let mut helps = [0usize; 8];
        let mut count = 0;
        let mut total_improves = 0;
        let mut readout_sign_agrees = 0;
        for &q in split {
            let Some(t) = labels[q] else {
                continue;
            };
            // Candidate labels come from the already frozen training routes.
            let c = (0..97)
                .filter(|&c| c != t)
                .max_by(|&a, &b| old[q * 97 + a].total_cmp(&old[q * 97 + b]))
                .unwrap();
            let terms: Vec<_> = contributions
                .iter()
                .map(|s| s[q * 97 + t] - s[q * 97 + c])
                .collect();
            let total: f64 = terms.iter().sum();
            count += 1;
            total_improves += usize::from(total > 0.);
            readout_sign_agrees += usize::from((terms[1] > 0.) == (total > 0.));
            for i in 0..8 {
                sums[i] += terms[i];
                helps[i] += usize::from(terms[i] > 0.);
            }
            records.push(json!([q, t, c, terms, total]));
        }
        result[name] = json!({"count":count,"mean_margin_contributions":sums.map(|x|x/count as f64),
            "positive_fraction":helps.map(|x|x as f64/count as f64),"total_positive_fraction":total_improves as f64/count as f64,
            "readout_credit_sign_agreement":readout_sign_agrees as f64/count as f64});
    }
    result["query_fields"] = json!([
        "query",
        "candidate_target",
        "old_worst_competitor",
        "margin_contributions",
        "total"
    ]);
    result["queries"] = json!(records);
    Ok(result)
}

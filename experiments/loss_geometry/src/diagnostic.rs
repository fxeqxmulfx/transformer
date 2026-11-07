//! LG1-LG4: probability geometry implied by observed CE, not unseen-data prediction.
use crate::{
    data::Split,
    model::{Forward, P, W},
};
use serde_json::{Value, json};
use std::collections::BTreeMap;

const TOLERANCE: f64 = 1e-10;
const METRICS: [&str; 8] = [
    "mean_cross_entropy",
    "missing_target_probability",
    "probability_squared_error",
    "probability_scatter",
    "probability_center_squared_error",
    "raw_state_scatter",
    "normalized_state_scatter",
    "raw_contrast_scatter",
];

fn target(q: usize) -> usize {
    (q / P + q % P).pow(3) % P
}
fn winner(row: &[f64]) -> usize {
    (1..row.len()).fold(0, |best, c| if row[c] > row[best] { c } else { best })
}

fn probability_row(logits: &[f64], label: usize) -> (Vec<f64>, f64, f64) {
    assert!(!logits.is_empty() && label < logits.len());
    assert!(logits.iter().all(|x| x.is_finite()));
    let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut p: Vec<_> = logits.iter().map(|x| (x - maximum).exp()).collect();
    let sum = p.iter().sum::<f64>();
    let ce = sum.ln() + (maximum - logits[label]);
    for v in &mut p {
        *v /= sum;
    }
    let error = p
        .iter()
        .enumerate()
        .map(|(c, &v)| (v - f64::from(c == label)).powi(2))
        .sum();
    (p, ce, error)
}

/// Arithmetic mean and average squared distance, computed independently in two passes.
fn moments(indices: &[usize], width: usize, f: impl Fn(usize, usize) -> f64) -> (Vec<f64>, f64) {
    assert!(!indices.is_empty() && width > 0);
    let mut center = vec![0.; width];
    for &q in indices {
        for (c, v) in center.iter_mut().enumerate() {
            *v += f(q, c) / indices.len() as f64;
        }
    }
    let mut scatter = 0.;
    for &q in indices {
        for (c, &v) in center.iter().enumerate() {
            scatter += (f(q, c) - v).powi(2);
        }
    }
    (center, scatter / indices.len() as f64)
}

pub fn analyze(out: &Forward, split: &Split) -> Value {
    let mut probabilities = Vec::with_capacity(P * P * P);
    let mut losses = Vec::new();
    let mut errors = Vec::new();
    let mut missing_violation = 0f64;
    let mut error_violation = 0f64;
    let mut mass_error = 0f64;
    for q in 0..P * P {
        let (p, ce, error) = probability_row(&out.logits[q * P..(q + 1) * P], target(q));
        mass_error = mass_error.max((p.iter().sum::<f64>() - 1.).abs());
        missing_violation = missing_violation.max(1. - p[target(q)] - ce);
        error_violation = error_violation.max(error - 2. * ce);
        probabilities.extend(p);
        losses.push(ce);
        errors.push(error);
    }
    assert!(mass_error < TOLERANCE && missing_violation < TOLERANCE && error_violation < TOLERANCE);
    let common = |indices: &[usize]| {
        json!({"count":indices.len(),
        "source_correct":indices.iter().filter(|&&q|winner(&out.logits[q*P..(q+1)*P])==target(q)).count(),
        "residual_only_correct":indices.iter().filter(|&&q|winner(&out.residual[q*P..(q+1)*P])==target(q)).count()})
    };
    let part = |indices: &[usize], orbit: bool| {
        let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for &q in indices {
            let s = (q / P + q % P) % P;
            let key = if orbit {
                s.min(35 * s % P).min(61 * s % P)
            } else {
                s
            };
            groups.entry(key).or_default().push(q);
        }
        let mut records = Vec::new();
        let mut max_decomposition_error = 0f64;
        let mut max_scatter_violation = 0f64;
        for (key, members) in groups {
            let label = target(members[0]);
            assert!(members.iter().all(|&q| target(q) == label));
            let n = members.len() as f64;
            let ce = members.iter().map(|&q| losses[q]).sum::<f64>() / n;
            let brier = members.iter().map(|&q| errors[q]).sum::<f64>() / n;
            let (center, scatter) = moments(&members, P, |q, c| probabilities[q * P + c]);
            let center_error = center
                .iter()
                .enumerate()
                .map(|(c, &v)| (v - f64::from(c == label)).powi(2))
                .sum::<f64>();
            max_decomposition_error =
                max_decomposition_error.max((scatter + center_error - brier).abs());
            max_scatter_violation = max_scatter_violation.max(scatter - 2. * ce);
            let raw_state = moments(&members, W, |q, c| out.states[q * W + c]).1;
            let normalized_state = moments(&members, W, |q, c| {
                out.states[q * W + c] / out.denominators[q]
            })
            .1;
            let raw_contrast = moments(&members, P, |q, c| {
                out.raw[q * P + label] - out.raw[q * P + c]
            })
            .1 / (P - 1) as f64;
            let normalized_contrast = moments(&members, P, |q, c| {
                out.logits[q * P + label] - out.logits[q * P + c]
            })
            .1 / (P - 1) as f64;
            records.push(json!({"key":key,"label":label,"count":members.len(),
                "source_correct":members.iter().filter(|&&q|winner(&out.logits[q*P..(q+1)*P])==label).count(),
                "mean_cross_entropy":ce,"missing_target_probability":1.-center[label],
                "probability_squared_error":brier,"probability_scatter":scatter,
                "probability_center_squared_error":center_error,"raw_state_scatter":raw_state,
                "normalized_state_scatter":normalized_state,"raw_contrast_scatter":raw_contrast,
                "normalized_contrast_scatter":normalized_contrast}));
        }
        assert!(max_decomposition_error < TOLERANCE && max_scatter_violation < TOLERANCE);
        let mut result = serde_json::Map::new();
        for key in METRICS
            .iter()
            .copied()
            .chain(["normalized_contrast_scatter"])
        {
            let value = records
                .iter()
                .map(|g| g[key].as_f64().unwrap() * g["count"].as_u64().unwrap() as f64)
                .sum::<f64>()
                / indices.len() as f64;
            assert!(value.is_finite());
            result.insert(key.to_owned(), json!(value));
        }
        result.insert("count".to_owned(), json!(indices.len()));
        result.insert(
            "max_decomposition_error".to_owned(),
            json!(max_decomposition_error),
        );
        result.insert(
            "max_scatter_bound_violation".to_owned(),
            json!(max_scatter_violation),
        );
        result.insert("groups".to_owned(), json!(records));
        Value::Object(result)
    };
    json!({"train":common(&split.train),"validation":common(&split.validation),
        "sum":{"train":part(&split.train,false),"validation":part(&split.validation,false)},
        "orbit":{"train":part(&split.train,true),"validation":part(&split.validation,true)},
        "row_audit":{"max_mass_error":mass_error,"max_missing_probability_violation":missing_violation,
        "max_probability_error_violation":error_violation},
        "status":"Observed loss bounds probability scatter on the same split; no generalization or monotonicity claim"})
}

pub fn transitions(cases: &[Value]) -> Vec<Value> {
    let mut results = Vec::new();
    for pair in cases.windows(2) {
        let (a, b) = (&pair[0], &pair[1]);
        if a["arm"] != b["arm"] || a["seed"] != b["seed"] {
            continue;
        }
        for grouping in ["sum", "orbit"] {
            for split in ["train", "validation"] {
                let mut deltas = serde_json::Map::new();
                for key in METRICS
                    .iter()
                    .copied()
                    .chain(["normalized_contrast_scatter"])
                {
                    deltas.insert(
                        key.to_owned(),
                        json!(
                            b["diagnostic"][grouping][split][key].as_f64().unwrap()
                                - a["diagnostic"][grouping][split][key].as_f64().unwrap()
                        ),
                    );
                }
                let loss_decreased = deltas["mean_cross_entropy"].as_f64().unwrap() < 0.;
                let raw_expanded = deltas["raw_contrast_scatter"].as_f64().unwrap() > 0.;
                results.push(json!({"arm":a["arm"],"seed":a["seed"],"from_step":a["step"],"to_step":b["step"],
                    "grouping":grouping,"split":split,"deltas":deltas,
                    "loss_decreased":loss_decreased,"loss_decreased_and_raw_contrast_expanded":loss_decreased && raw_expanded}));
            }
        }
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn known_probabilities_and_shift_invariance() {
        let (p, l, e) = probability_row(&[3f64.ln(), 0.], 0);
        assert!((p[0] - 0.75).abs() < 1e-14 && (p[1] - 0.25).abs() < 1e-14);
        assert!((l - (4f64 / 3.).ln()).abs() < 1e-14 && (e - 0.125).abs() < 1e-14);
        let (shifted, sl, se) = probability_row(&[3f64.ln() + 100., 100.], 0);
        assert!(
            (shifted[0] - p[0]).abs() < 1e-14 && (sl - l).abs() < 1e-14 && (se - e).abs() < 1e-14
        );
    }
    #[test]
    fn nonconstant_probability_centering_identity() {
        let p = [[0.75, 0.25], [0.25, 0.75]];
        let (center, scatter) = moments(&[0, 1], 2, |q, c| p[q][c]);
        assert_eq!(center, [0.5, 0.5]);
        assert_eq!(scatter, 0.125);
        let squared_error = p
            .iter()
            .map(|r| (r[0] - 1.).powi(2) + r[1].powi(2))
            .sum::<f64>()
            / 2.;
        assert_eq!(squared_error, scatter + 0.5);
    }
    #[test]
    fn actual_shared_gradient_lowers_loss_and_expands_raw_contrasts() {
        let loss = |t: f64| ((1. + (-t).exp()).ln() + (1. + (-2. * t).exp()).ln()) / 2.;
        for t in [0f64, 0.2, 1., 4.] {
            let g = (-1. / (1. + t.exp()) - 2. / (1. + (2. * t).exp())) / 2.;
            let fd = (loss(t + 1e-5) - loss(t - 1e-5)) / 2e-5;
            assert!((g - fd).abs() < 1e-9);
            for rate in [0.01, 0.1, 1.] {
                let next = t - rate * g;
                assert!(loss(next) < loss(t));
                let before = moments(&[0, 1], 1, |i, _| (i + 1) as f64 * t).1;
                let after = moments(&[0, 1], 1, |i, _| (i + 1) as f64 * next).1;
                assert!(before < after);
            }
        }
    }
}

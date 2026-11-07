//! PT1-PT4: train-only odds and unlabeled normalized states construct query bounds.
use crate::algebra::linear;
use serde_json::{Value, json};
use std::collections::BTreeMap;

pub const METHODS: [&str; 5] = [
    "center",
    "nearest",
    "best_anchor",
    "anchor_envelope",
    "center_anchor_envelope",
];
#[derive(Clone, Copy)]
pub enum Grouping {
    Sum,
    Orbit,
}

pub fn summary(values: &[f64]) -> Value {
    if values.is_empty() {
        return Value::Null;
    }
    assert!(values.iter().all(|v| v.is_finite()));
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    json!({"min":sorted[0],"median":sorted[sorted.len()/2],"max":sorted[sorted.len()-1],
        "mean":sorted.iter().sum::<f64>()/sorted.len() as f64})
}
pub fn log_sum_exp(values: &[f64]) -> f64 {
    assert!(!values.is_empty() && values.iter().all(|v| v.is_finite()));
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    maximum + values.iter().map(|v| (v - maximum).exp()).sum::<f64>().ln()
}
pub fn softplus(x: f64) -> f64 {
    if x > 0. {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

#[derive(Clone, Copy)]
pub struct Bound {
    pub log_odds: f64,
    pub loss: f64,
    pub probability: f64,
    pub gap: f64,
}
impl Bound {
    fn from_upper(upper: &[f64], label: usize) -> Self {
        let wrong: Vec<_> = upper
            .iter()
            .enumerate()
            .filter(|(c, _)| *c != label)
            .map(|(_, u)| *u)
            .collect();
        let log_odds = log_sum_exp(&wrong);
        let loss = softplus(log_odds);
        Self {
            log_odds,
            loss,
            probability: (-loss).exp(),
            gap: -wrong.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        }
    }
    fn frozen(self) -> Value {
        json!({"log_odds":self.log_odds,"loss_upper":self.loss,"probability_lower":self.probability,"gap_lower":self.gap})
    }
}
pub struct Constructed {
    pub labels: Vec<Option<usize>>,
    pub bounds: Vec<Vec<Option<Bound>>>,
    pub groups: Vec<Value>,
    nearest: Vec<Option<usize>>,
    best: Vec<Option<usize>>,
}
impl Constructed {
    pub fn frozen(&self) -> Value {
        json!({"labels":self.labels,"bounds":self.bounds.iter().map(|method|method.iter().map(|b|b.map(Bound::frozen)).collect::<Vec<_>>()).collect::<Vec<_>>(),
            "nearest":self.nearest,"best":self.best,"groups":self.groups,"methods":METHODS})
    }
}

fn distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Query labels/logits/probabilities do not cross this interface. No task cubing occurs here.
pub fn construct(
    inputs: &[(usize, usize)],
    states: &[f64],
    embedding: &[f64],
    training: &[(usize, usize)],
    width: usize,
    classes: usize,
    grouping: Grouping,
) -> Constructed {
    let n = inputs.len();
    assert!(
        width > 0 && classes > 1 && states.len() == n * width && embedding.len() == classes * width
    );
    assert!(states.iter().chain(embedding).all(|v| v.is_finite()));
    assert!(inputs.iter().all(|&(a, b)| a < 97 && b < 97));
    let mut observed = vec![None; n];
    for &(q, label) in training {
        assert!(q < n && label < classes && observed[q].is_none());
        observed[q] = Some(label);
    }
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (q, &(a, b)) in inputs.iter().enumerate() {
        let s = (a + b) % 97;
        let key = match grouping {
            Grouping::Sum => s,
            Grouping::Orbit => s.min(35 * s % 97).min(61 * s % 97),
        };
        groups.entry(key).or_default().push(q);
    }
    let mut result = Constructed {
        labels: vec![None; n],
        bounds: vec![vec![None; n]; METHODS.len()],
        groups: Vec::new(),
        nearest: vec![None; n],
        best: vec![None; n],
    };
    for (key, queries) in groups {
        let train: Vec<_> = queries
            .iter()
            .copied()
            .filter(|&q| observed[q].is_some())
            .collect();
        if train.is_empty() {
            result.groups.push(
                json!({"key":key,"training_count":0,"query_count":queries.len(),"abstained":true}),
            );
            continue;
        }
        let first = train[0];
        let label = observed[first].unwrap();
        assert!(
            train.iter().all(|&q| observed[q] == Some(label)),
            "inconsistent training labels"
        );
        let anchor_sum = (inputs[first].0 + inputs[first].1) % 97;
        for &q in &queries {
            let sum = (inputs[q].0 + inputs[q].1) % 97;
            let allowed: &[usize] = match grouping {
                Grouping::Sum => &[1],
                Grouping::Orbit => &[1, 35, 61],
            };
            let u = allowed
                .iter()
                .copied()
                .find(|u| u * anchor_sum % 97 == sum)
                .expect("no task-preserving route");
            let shift = (inputs[q].0 + 97 - u * inputs[first].0 % 97) % 97;
            assert_eq!((u * inputs[first].0 + shift) % 97, inputs[q].0);
            assert_eq!((u * inputs[first].1 + 97 - shift) % 97, inputs[q].1);
        }
        let nt = train.len();
        let train_states: Vec<_> = train
            .iter()
            .flat_map(|&q| states[q * width..(q + 1) * width].iter().copied())
            .collect();
        // Only training rows are read out. Query rows are used solely in state distances.
        let logits = linear(&train_states, nt, width, embedding, classes);
        let odds: Vec<_> = (0..nt)
            .flat_map(|i| {
                (0..classes).map({
                    let logits = &logits;
                    move |c| logits[i * classes + c] - logits[i * classes + label]
                })
            })
            .collect();
        let mut state_sum = vec![0.; width];
        let mut odds_sum = vec![0.; classes];
        for i in 0..nt {
            for k in 0..width {
                state_sum[k] += train_states[i * width + k];
            }
            for c in 0..classes {
                odds_sum[c] += odds[i * classes + c];
            }
        }
        let contrast_norms: Vec<_> = (0..classes)
            .map(|c| {
                distance(
                    &embedding[c * width..(c + 1) * width],
                    &embedding[label * width..(label + 1) * width],
                )
            })
            .collect();
        let training_losses: Vec<_> = (0..nt)
            .map(|i| Bound::from_upper(&odds[i * classes..(i + 1) * classes], label).loss)
            .collect();
        let mut nearest_distances = Vec::new();
        let mut abstentions = 0;
        for &q in &queries {
            let own = train.iter().position(|&i| i == q);
            let remaining = nt - usize::from(own.is_some());
            if remaining == 0 {
                abstentions += 1;
                continue;
            }
            result.labels[q] = Some(label);
            let center: Vec<_> = (0..width)
                .map(|k| {
                    (state_sum[k] - own.map_or(0., |i| train_states[i * width + k]))
                        / remaining as f64
                })
                .collect();
            let center_distance = distance(&states[q * width..(q + 1) * width], &center);
            let center_upper: Vec<_> = (0..classes)
                .map(|c| {
                    (odds_sum[c] - own.map_or(0., |i| odds[i * classes + c])) / remaining as f64
                        + center_distance * contrast_norms[c]
                })
                .collect();
            result.bounds[0][q] = Some(Bound::from_upper(&center_upper, label));
            let mut upper = vec![0.; classes];
            let mut envelope = vec![f64::INFINITY; classes];
            let mut nearest = f64::INFINITY;
            let mut best = f64::INFINITY;
            for (i, &anchor) in train.iter().enumerate() {
                if anchor == q {
                    continue;
                }
                let d = distance(
                    &states[q * width..(q + 1) * width],
                    &train_states[i * width..(i + 1) * width],
                );
                for c in 0..classes {
                    upper[c] = odds[i * classes + c] + d * contrast_norms[c];
                    envelope[c] = envelope[c].min(upper[c]);
                }
                let bound = Bound::from_upper(&upper, label);
                if d < nearest {
                    nearest = d;
                    result.nearest[q] = Some(anchor);
                    result.bounds[1][q] = Some(bound);
                }
                if bound.log_odds < best {
                    best = bound.log_odds;
                    result.best[q] = Some(anchor);
                    result.bounds[2][q] = Some(bound);
                }
            }
            result.bounds[3][q] = Some(Bound::from_upper(&envelope, label));
            for c in 0..classes {
                envelope[c] = envelope[c].min(center_upper[c]);
            }
            result.bounds[4][q] = Some(Bound::from_upper(&envelope, label));
            assert!(result.bounds[3][q].unwrap().log_odds <= best + 1e-10);
            assert!(
                result.bounds[4][q].unwrap().log_odds
                    <= result.bounds[0][q].unwrap().log_odds + 1e-10
            );
            nearest_distances.push(nearest);
        }
        result.groups.push(json!({"key":key,"label":label,"training_count":nt,"query_count":queries.len(),
            "self_exclusion_abstentions":abstentions,"routes_verified":true,"training_labels_consistent":true,
            "training_cross_entropy":summary(&training_losses),"nearest_distance":summary(&nearest_distances)}));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn known_odds_bound_and_large_log_odds_are_stable() {
        let b = Bound::from_upper(&[0., (0.25f64).ln()], 0);
        assert!((b.probability - 0.8).abs() < 1e-14 && (b.loss - 1.25f64.ln()).abs() < 1e-14);
        let big = Bound::from_upper(&[0., 1000.], 0);
        assert!(big.loss.is_finite() && big.probability == 0. && big.log_odds == 1000.);
    }
    #[test]
    fn nonzero_state_distance_transfers_without_query_labels() {
        let c = construct(
            &[(0, 1), (1, 0), (2, 96)],
            &[2., 1., 2., -1., 1.5, 0.],
            &[-1., 0., 1., 0., 0., 0.],
            &[(0, 1), (1, 1)],
            2,
            3,
            Grouping::Orbit,
        );
        let actual = 1. / (1. + (-3f64).exp() + (-1.5f64).exp());
        assert_eq!(c.labels[2], Some(1));
        for m in &c.bounds {
            let b = m[2].unwrap();
            assert!(b.probability > 0.5 && b.probability <= actual + 1e-14);
        }
        assert!((c.bounds[0][2].unwrap().probability - actual).abs() < 1e-14);
        assert!(c.groups[0]["nearest_distance"]["min"].as_f64().unwrap() > 1.);
    }
    #[test]
    fn self_anchor_is_excluded_and_unobserved_orbits_abstain() {
        let c = construct(
            &[(0, 1), (0, 0)],
            &[2., 0., 2., 0.],
            &[-1., 0., 1., 0.],
            &[(0, 1)],
            2,
            2,
            Grouping::Sum,
        );
        assert_eq!(c.labels, [None, None]);
        assert!(c.bounds.iter().all(|m| m.iter().all(Option::is_none)));
    }
}

//! ST1-ST5: fixed ranks and label-preserving groups; no query target/score argument.
use crate::reconstruction;
use serde_json::{Value, json};
use std::collections::BTreeMap;

pub const RANKS: [usize; 9] = [0, 1, 2, 4, 8, 16, 32, 64, 128];
pub fn methods() -> Vec<String> {
    RANKS
        .iter()
        .flat_map(|r| [format!("signed_r{r}"), format!("dual_r{r}")])
        .collect()
}
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
    pub fn from_upper(upper: &[f64], label: usize) -> Self {
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
    pub methods: Vec<String>,
    pub groups: Vec<Value>,
}
impl Constructed {
    pub fn frozen(&self) -> Value {
        json!({"labels":self.labels,"bounds":self.bounds.iter().map(|method|method.iter().map(|b|b.map(Bound::frozen)).collect::<Vec<_>>()).collect::<Vec<_>>(),
            "methods":self.methods,"groups":self.groups})
    }
}

/// Geometry and observed labels only; training queries use opposite fixed folds.
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
        bounds: vec![vec![None; n]; RANKS.len() * 2],
        methods: methods(),
        groups: Vec::new(),
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
        let mut folds = Vec::new();
        for fold in 0..3 {
            let references: Vec<_> = train
                .iter()
                .enumerate()
                .filter(|(i, _)| fold == 0 || i % 2 == fold - 1)
                .map(|(_, &q)| q)
                .collect();
            let selected: Vec<_> = if fold == 0 {
                queries
                    .iter()
                    .copied()
                    .filter(|&q| observed[q].is_none())
                    .collect()
            } else {
                train
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| i % 2 != fold - 1)
                    .map(|(_, &q)| q)
                    .collect()
            };
            if selected.is_empty() {
                continue;
            }
            assert!(
                selected.iter().all(|q| !references.contains(q)),
                "self reference entered a fold"
            );
            if references.is_empty() {
                folds.push(json!({"fold":fold,"training_count":0,"query_count":selected.len(),"abstained":true}));
                continue;
            }
            let (bounds, stats) = reconstruction::evaluate(
                states,
                embedding,
                &references,
                &selected,
                width,
                classes,
                label,
            );
            for (i, &q) in selected.iter().enumerate() {
                result.labels[q] = Some(label);
                for (m, bound) in bounds.iter().enumerate() {
                    result.bounds[m][q] = Some(bound[i]);
                }
            }
            folds.push(json!({"fold":fold,"reference_indices":references,"query_count":selected.len(),"geometry":stats}));
        }
        result.groups.push(
            json!({"key":key,"label":label,"training_count":train.len(),"query_count":queries.len(),
            "routes_verified":true,"training_labels_consistent":true,"folds":folds}),
        );
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn signed_extrapolation_retains_a_nonzero_directional_residual() {
        let c = construct(
            &[(0, 1), (1, 0), (2, 96)],
            &[0.5, 0., 1.5, 0., 3., 0.25],
            &[2., 0., 0., 1.],
            &[(0, 0), (1, 0)],
            2,
            2,
            Grouping::Orbit,
        );
        let actual = 1. / (1. + (-5.75f64).exp());
        assert_eq!(c.labels[2], Some(0));
        assert!(c.bounds[0][2].unwrap().gap < 0.);
        let signed = c.bounds[2][2].unwrap();
        let dual = c.bounds[3][2].unwrap();
        assert!(signed.probability > 0.5 && dual.probability > signed.probability);
        assert!((dual.probability - actual).abs() < 1e-13);
        assert!((dual.gap - 5.75).abs() < 1e-13);
        for m in &c.bounds {
            assert!(m[2].unwrap().probability <= actual + 1e-13);
        }
    }
    #[test]
    fn missing_orbits_and_singleton_training_folds_abstain() {
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
    #[test]
    fn known_odds_and_large_log_odds_are_stable() {
        let b = Bound::from_upper(&[0., (0.25f64).ln()], 0);
        assert!((b.probability - 0.8).abs() < 1e-14);
        assert_eq!(Bound::from_upper(&[0., 1000.], 0).loss, 1000.);
    }
}

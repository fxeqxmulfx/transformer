//! Fixed cubic routes and opposite training folds from signed_transport at
//! 07c328f1. Constructor has no query labels, future states or future scores.
use crate::reconstruction::Family;
use crate::reconstruction::{self, RANKS};
use serde_json::{Value, json};
use std::collections::BTreeMap;
pub struct Geometry {
    pub labels: Vec<Option<usize>>,
    pub upper: Vec<Vec<f64>>,
    pub groups: Vec<Value>,
}
pub fn construct(
    inputs: &[(usize, usize)],
    states: &[f64],
    embedding: &[f64],
    training: &[(usize, usize)],
    width: usize,
    classes: usize,
    family: Family,
) -> Geometry {
    let n = inputs.len();
    assert_eq!(states.len(), n * width);
    assert_eq!(embedding.len(), classes * width);
    let mut observed = vec![None; n];
    for &(q, t) in training {
        assert!(q < n && t < classes && observed[q].is_none());
        observed[q] = Some(t);
    }
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (q, &(a, b)) in inputs.iter().enumerate() {
        let s = (a + b) % 97;
        groups
            .entry(s.min(35 * s % 97).min(61 * s % 97))
            .or_default()
            .push(q);
    }
    let mut result = Geometry {
        labels: vec![None; n],
        upper: vec![vec![f64::INFINITY; n * classes]; RANKS.len()],
        groups: vec![],
    };
    for (key, queries) in groups {
        let train: Vec<_> = queries
            .iter()
            .copied()
            .filter(|&q| observed[q].is_some())
            .collect();
        if train.is_empty() {
            result.groups.push(json!({"key":key,"abstained":true}));
            continue;
        }
        let first = train[0];
        let label = observed[first].unwrap();
        assert!(
            train.iter().all(|&q| observed[q] == Some(label)),
            "inconsistent observed labels"
        );
        let anchor = (inputs[first].0 + inputs[first].1) % 97;
        for &q in &queries {
            let sum = (inputs[q].0 + inputs[q].1) % 97;
            let u = [1, 35, 61]
                .into_iter()
                .find(|u| u * anchor % 97 == sum)
                .expect("missing cubic route");
            let shift = (inputs[q].0 + 97 - u * inputs[first].0 % 97) % 97;
            assert_eq!((u * inputs[first].0 + shift) % 97, inputs[q].0);
            assert_eq!((u * inputs[first].1 + 97 - shift) % 97, inputs[q].1);
        }
        let mut folds = vec![];
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
            if references.is_empty() {
                folds.push(json!({"fold":fold,"abstained":true}));
                continue;
            }
            assert!(
                selected.iter().all(|q| !references.contains(q)),
                "self reference"
            );
            let (bounds, stats) = reconstruction::evaluate(reconstruction::Problem {
                family,
                states,
                embedding,
                references: &references,
                queries: &selected,
                width,
                classes,
                label,
            });
            for (i, &q) in selected.iter().enumerate() {
                result.labels[q] = Some(label);
                for (out, bound) in result.upper.iter_mut().zip(&bounds) {
                    out[q * classes..(q + 1) * classes]
                        .copy_from_slice(&bound[i * classes..(i + 1) * classes]);
                }
            }
            folds.push(
                json!({"fold":fold,"references":references,"queries":selected,"geometry":stats}),
            );
        }
        result
            .groups
            .push(json!({"key":key,"label":label,"routes_verified":true,"folds":folds}));
    }
    result
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn extrapolation_keeps_nonzero_residual_and_unseen_labels_are_absent() {
        let g = construct(
            &[(0, 1), (1, 0), (2, 96)],
            &[0.5, 0., 1.5, 0., 3., 0.25],
            &[2., 0., 0., 1.],
            &[(0, 0), (1, 0)],
            2,
            2,
            Family::Hidden,
        );
        assert_eq!(g.labels[2], Some(0));
        assert!((g.upper[1][5] + 5.75).abs() < 1e-12);
        assert!(g.upper[0][5] > g.upper[1][5]);
    }
    #[test]
    fn missing_groups_and_singleton_training_folds_abstain() {
        let g = construct(
            &[(0, 1), (0, 0)],
            &[2., 0., 1., 0.],
            &[2., 0., 0., 1.],
            &[(0, 0)],
            2,
            2,
            Family::Hidden,
        );
        assert_eq!(g.labels, vec![None, None]);
    }
}

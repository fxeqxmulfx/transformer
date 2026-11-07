//! TT1-TT4: constructor has no query-label, query-logit or query-error argument.
use crate::algebra::{linear, solve_spd};
use serde_json::{Value, json};
use std::collections::BTreeMap;

pub const FACTORS: [f64; 4] = [1e-4, 1e-2, 1., 100.];
pub const METHODS: [&str; 7] = [
    "euclidean",
    "ridge_1e-4",
    "ridge_1e-2",
    "ridge_1",
    "ridge_100",
    "complete_choice",
    "competitor_choice",
];

#[derive(Clone, Copy)]
pub enum Grouping {
    Sum,
    Orbit,
}

fn key(input: (usize, usize), grouping: Grouping) -> usize {
    let s = (input.0 + input.1) % 97;
    match grouping {
        Grouping::Sum => s,
        Grouping::Orbit => s.min(35 * s % 97).min(61 * s % 97),
    }
}

pub fn summary(values: &[f64]) -> Value {
    if values.is_empty() {
        return Value::Null;
    }
    assert!(values.iter().all(|x| x.is_finite()));
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    json!({"min":v[0],"median":v[v.len()/2],"max":v[v.len()-1],"mean":v.iter().sum::<f64>()/v.len() as f64})
}

pub struct Constructed {
    pub labels: Vec<Option<usize>>,
    pub lower: Vec<Vec<Option<f64>>>,
    pub groups: Vec<Value>,
}

impl Constructed {
    pub fn frozen(&self) -> Value {
        json!({"labels":self.labels,"lower_bounds":self.lower,"groups":self.groups,"methods":METHODS})
    }
}

/// Only training labels cross this interface. All query data are unlabeled states/operands.
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
    assert!(width > 0 && classes > 1);
    assert_eq!(states.len(), n * width);
    assert_eq!(embedding.len(), classes * width);
    assert!(states.iter().chain(embedding).all(|x| x.is_finite()));
    assert!(inputs.iter().all(|&(a, b)| a < 97 && b < 97));
    let mut observed = vec![None; n];
    for &(q, label) in training {
        assert!(q < n && label < classes && observed[q].is_none());
        observed[q] = Some(label);
    }
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (q, &input) in inputs.iter().enumerate() {
        groups.entry(key(input, grouping)).or_default().push(q);
    }
    let mut result = Constructed {
        labels: vec![None; n],
        lower: vec![vec![None; n]; METHODS.len()],
        groups: Vec::new(),
    };
    for (group, queries) in groups {
        let train: Vec<_> = queries
            .iter()
            .copied()
            .filter(|&q| observed[q].is_some())
            .collect();
        if train.is_empty() {
            result.groups.push(json!({"key":group,"training_count":0,"query_count":queries.len(),"abstained":true}));
            continue;
        }
        let anchor = train[0];
        let label = observed[anchor].unwrap();
        assert!(
            train.iter().all(|&q| observed[q] == Some(label)),
            "inconsistent observed labels"
        );
        let anchor_sum = (inputs[anchor].0 + inputs[anchor].1) % 97;
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
                .expect("no label-preserving route");
            let shift = (inputs[q].0 + 97 - (u * inputs[anchor].0) % 97) % 97;
            assert_eq!((u * inputs[anchor].0 + shift) % 97, inputs[q].0);
            assert_eq!((u * inputs[anchor].1 + 97 - shift) % 97, inputs[q].1);
            result.labels[q] = Some(label);
        }
        let nt = train.len();
        let nq = queries.len();
        let mut center = vec![0.; width];
        for &q in &train {
            for k in 0..width {
                center[k] += states[q * width + k] / nt as f64;
            }
        }
        let centered: Vec<_> = train
            .iter()
            .flat_map(|&q| {
                (0..width).map({
                    let center = &center;
                    move |k| states[q * width + k] - center[k]
                })
            })
            .collect();
        let displacements: Vec<_> = queries
            .iter()
            .flat_map(|&q| {
                (0..width).map({
                    let center = &center;
                    move |k| states[q * width + k] - center[k]
                })
            })
            .collect();
        let contrasts: Vec<_> = (0..classes)
            .flat_map(|v| {
                (0..width).map(move |k| embedding[label * width + k] - embedding[v * width + k])
            })
            .collect();
        let center_gaps = linear(&center, 1, width, &contrasts, classes);
        let deviations = linear(&centered, nt, width, &contrasts, classes);
        let variance: Vec<_> = (0..classes)
            .map(|v| {
                (0..nt)
                    .map(|i| deviations[i * classes + v].powi(2))
                    .sum::<f64>()
            })
            .collect();
        let norm_sq: Vec<_> = contrasts
            .chunks_exact(width)
            .map(|row| row.iter().map(|x| x * x).sum::<f64>())
            .collect();
        let transpose: Vec<_> = (0..width)
            .flat_map(|k| {
                (0..nt).map({
                    let centered = &centered;
                    move |i| centered[i * width + k]
                })
            })
            .collect();
        let gram = linear(&transpose, width, nt, &transpose, width);
        let trace = (0..width).map(|k| gram[k * width + k]).sum::<f64>();
        let scale = (trace / width as f64).max(1e-12);
        let mut best_by_competitor = vec![f64::NEG_INFINITY; nq * classes];
        for (i, &q) in queries.iter().enumerate() {
            let energy = displacements[i * width..(i + 1) * width]
                .iter()
                .map(|x| x * x)
                .sum::<f64>();
            let mut lower = f64::INFINITY;
            for v in 0..classes {
                if v == label {
                    continue;
                }
                let bound = center_gaps[v] - (energy * norm_sq[v]).sqrt();
                lower = lower.min(bound);
                best_by_competitor[i * classes + v] = bound;
            }
            result.lower[0][q] = Some(lower);
        }
        let mut ridge_reports = Vec::new();
        for (method, factor) in FACTORS.iter().enumerate() {
            let lambda = factor * scale;
            assert!(lambda > 0. && lambda.is_finite());
            let mut regularized = gram.clone();
            for k in 0..width {
                regularized[k * width + k] += lambda;
            }
            let solution = solve_spd(&regularized, &displacements, width);
            let coefficients = linear(&solution, nq, width, &centered, nt);
            let reconstructed = linear(&coefficients, nq, nt, &transpose, width);
            let mut energies = Vec::new();
            let mut residual_energies = Vec::new();
            let mut solve_error = 0f64;
            for (i, &q) in queries.iter().enumerate() {
                let coefficient_energy = coefficients[i * nt..(i + 1) * nt]
                    .iter()
                    .map(|x| x * x)
                    .sum::<f64>();
                let mut residual_energy = 0.;
                for k in 0..width {
                    let r = displacements[i * width + k] - reconstructed[i * width + k];
                    residual_energy += r * r;
                    solve_error = solve_error.max((r - lambda * solution[i * width + k]).abs());
                }
                let energy = coefficient_energy + residual_energy / lambda;
                assert!(energy.is_finite() && energy >= 0.);
                energies.push(energy);
                residual_energies.push(residual_energy);
                let mut lower = f64::INFINITY;
                for v in 0..classes {
                    if v == label {
                        continue;
                    }
                    let bound =
                        center_gaps[v] - (energy * (variance[v] + lambda * norm_sq[v])).sqrt();
                    lower = lower.min(bound);
                    best_by_competitor[i * classes + v] =
                        best_by_competitor[i * classes + v].max(bound);
                }
                result.lower[method + 1][q] = Some(lower);
            }
            ridge_reports.push(json!({"factor":factor,"lambda":lambda,"transport_energy":summary(&energies),
                "residual_squared_norm":summary(&residual_energies),"normal_equation_max_error":solve_error}));
        }
        for (i, &q) in queries.iter().enumerate() {
            let complete = (0..5)
                .map(|m| result.lower[m][q].unwrap())
                .fold(f64::NEG_INFINITY, f64::max);
            let mixed = (0..classes)
                .filter(|&v| v != label)
                .map(|v| best_by_competitor[i * classes + v])
                .fold(f64::INFINITY, f64::min);
            assert!(complete <= mixed + 1e-10 && mixed.is_finite());
            result.lower[5][q] = Some(complete);
            result.lower[6][q] = Some(mixed);
        }
        result.groups.push(json!({"key":group,"anchor":anchor,"label":label,"training_count":nt,
            "query_count":nq,"routes_verified":true,"training_labels_consistent":true,"covariance_trace":trace,
            "center_min_gap":center_gaps.iter().enumerate().filter(|(v,_)|*v!=label).map(|(_,g)|*g).fold(f64::INFINITY,f64::min),
            "training_contrast_variation":summary(&variance),"ridges":ridge_reports}));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nonzero_residual_and_no_query_label_still_certify() {
        let c = construct(
            &[(0, 1), (1, 0), (2, 96)],
            &[1., 1., 1., -1., 1.1, 0.],
            &[-1., 0., 1., 0., 0., 0.],
            &[(0, 1), (1, 1)],
            2,
            3,
            Grouping::Orbit,
        );
        assert_eq!(c.labels[2], Some(1));
        for m in 0..METHODS.len() {
            assert!(c.lower[m][2].unwrap() > 0.89);
        }
        assert!(
            c.groups[0]["ridges"][0]["residual_squared_norm"]["max"]
                .as_f64()
                .unwrap()
                > 0.009
        );
    }
    #[test]
    fn unseen_direction_requires_residual_penalty() {
        let c = construct(
            &[(0, 1), (1, 0), (2, 96)],
            &[1., 0., 1., 0., -1., 0.],
            &[-1., 0., 1., 0., 0., 0.],
            &[(0, 1), (1, 1)],
            2,
            3,
            Grouping::Orbit,
        );
        assert_eq!(c.labels[2], Some(1));
        assert!(c.lower.iter().all(|m| m[2].unwrap() < 0.));
    }
    #[test]
    fn absent_training_orbit_abstains() {
        let c = construct(
            &[(0, 1), (0, 0)],
            &[1., 0., 1., 0.],
            &[-1., 0., 1., 0., 0., 0.],
            &[(0, 1)],
            2,
            3,
            Grouping::Sum,
        );
        assert_eq!(c.labels[1], None);
        assert!(c.lower.iter().all(|m| m[1].is_none()));
    }
}

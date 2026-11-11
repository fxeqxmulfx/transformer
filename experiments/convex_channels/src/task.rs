//! Frozen categorical tasks and diagnostics. Label rules exist only here;
//! the Channel forward and optimizer never receive the rule table.

use crate::channel::Channel;
use convex_recall_compare::{
    data::{self, Dataset, Sample},
    mt::Mt,
};
use serde_json::{Value, json};

pub const KEYS: usize = 4;
pub const VALUES: usize = 8;
pub const CLASSES: usize = KEYS + 2 * VALUES;
pub const TRAIN: usize = 256;
pub const TEST: usize = 256;

pub struct Task {
    pub data: Dataset,
    pub rule: Vec<usize>,
    pub output_start: usize,
}

pub fn make(seed: u32, recoded: bool) -> Task {
    let mut data = data::mqar(seed, KEYS, VALUES, TRAIN, TEST);
    data.vocab = CLASSES;
    let output_start = if recoded { KEYS + VALUES } else { KEYS };
    let mut rng = Mt::new(seed.wrapping_add(75_019));
    let mut rule = Vec::new();
    for _ in 0..KEYS {
        let mut permutation: Vec<_> = (0..VALUES).collect();
        if recoded {
            rng.shuffle(&mut permutation);
        }
        rule.extend(permutation.iter().map(|v| output_start + v));
    }
    for s in data.train.iter_mut().chain(&mut data.test) {
        s.targets = s.queries.iter().map(|&q| rule[q * VALUES + s.values[q]]).collect();
    }
    Task { data, rule, output_start }
}

/// Checks precisely the CC5 atom-separation premise on the training data.
pub fn separates_atoms(task: &Task) -> bool {
    let shape = Channel::new(KEYS, KEYS, VALUES, CLASSES);
    let mut contradicted = vec![false; shape.weights.len()];
    for s in &task.data.train {
        for (&q, &target) in s.queries.iter().zip(&s.targets) {
            assert_eq!(target, task.rule[q * VALUES + s.values[q]]);
            for (k, &v) in s.values.iter().enumerate() {
                for c in 0..CLASSES {
                    if c != target {
                        contradicted[shape.index(q, k, v, c)] = true;
                    }
                }
            }
        }
    }
    (0..KEYS).all(|q| {
        (0..KEYS).all(|k| {
            (0..VALUES).all(|v| {
                (0..CLASSES).all(|c| {
                    (k == q && c == task.rule[q * VALUES + v])
                        || contradicted[shape.index(q, k, v, c)]
                })
            })
        })
    })
}

/// Every possible dictionary, used only for the finite spectral diagnostic.
pub fn all_dictionaries(task: &Task) -> Vec<Sample> {
    (0..VALUES.pow(KEYS as u32))
        .map(|id| {
            let mut index = id;
            let values: Vec<_> = (0..KEYS)
                .map(|_| {
                    let v = index % VALUES;
                    index /= VALUES;
                    v
                })
                .collect();
            let queries: Vec<_> = (0..KEYS).collect();
            let mut tokens = Vec::new();
            for (k, &v) in values.iter().enumerate() {
                tokens.extend([k, KEYS + v]);
            }
            tokens.extend(&queries);
            let targets = queries.iter().map(|&q| task.rule[q * VALUES + values[q]]).collect();
            Sample { tokens, targets, values, queries, id }
        })
        .collect()
}

fn sign(task: &Task, class: usize) -> f64 {
    if (task.output_start..task.output_start + VALUES).contains(&class) {
        if class - task.output_start < VALUES / 2 { 1.0 } else { -1.0 }
    } else {
        0.0
    }
}

/// Compute the target Gram and error of explicit rank-d row truncations.
/// This is an exact finite diagnostic of the R3 bound, not a training feature.
pub fn spectrum(task: &Task, channel: Option<&Channel>) -> Value {
    let samples = all_dictionaries(task);
    let mut gram = vec![vec![0i64; KEYS]; KEYS];
    let mut rank_errors = vec![0.0; KEYS + 1];
    let mut model_error = 0.0;
    for s in &samples {
        let target: Vec<_> = s.targets.iter().map(|&t| sign(task, t)).collect();
        for q in 0..KEYS {
            for r in 0..KEYS {
                gram[q][r] += (target[q] * target[r]) as i64;
            }
        }
        for (d, error) in rank_errors.iter_mut().enumerate() {
            *error += target[d..].iter().map(|x| x * x).sum::<f64>();
        }
        if let Some(ch) = channel {
            let logits = ch.logits(s);
            for q in 0..KEYS {
                let scalar: f64 =
                    (0..CLASSES).map(|c| logits[q * CLASSES + c] * sign(task, c)).sum();
                model_error += (scalar - target[q]).powi(2);
            }
        }
    }
    for q in 0..KEYS {
        for r in 0..KEYS {
            assert_eq!(gram[q][r], if q == r { samples.len() as i64 } else { 0 });
        }
    }
    for (d, &error) in rank_errors.iter().enumerate() {
        assert_eq!(error, ((KEYS - d) * samples.len()) as f64);
    }
    json!({"rank":KEYS,"squared_singular_value":samples.len(),"gram":gram,
        "categorical_feature_count":KEYS*VALUES,"rank_d_truncation_errors":rank_errors,
        "model_frobenius_error_sq":channel.map(|_| model_error),
        "model_relative_error_sq":channel.map(|_| model_error / (KEYS * samples.len()) as f64)})
}

/// Distance to the realizable parameter witness, used only in reporting.
pub fn rule_error(channel: &Channel, task: &Task) -> f64 {
    let mut error: f64 = 0.0;
    for q in 0..KEYS {
        for k in 0..KEYS {
            for v in 0..VALUES {
                for c in 0..CLASSES {
                    let target = f64::from(k == q && c == task.rule[q * VALUES + v]);
                    error = error.max((channel.weights[channel.index(q, k, v, c)] - target).abs());
                }
            }
        }
    }
    error
}

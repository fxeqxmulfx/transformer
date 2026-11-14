//! Bind the unchanged CFt4/CFt5 component using training prefixes only.
//! No task identity, generator parameter, held-out label or answer rule enters.
use crate::{PROTOCOL, Result, complex, data, metrics};
use complex::{Input, Observation, binding, features, model::Model};
use serde_json::{Value, json};
use std::collections::BTreeMap;

pub fn request(vocabulary: usize, rows: &[data::Example]) -> Result<binding::Request> {
    if rows.is_empty() || rows.len() > 20_000 {
        return Err("empty or oversized training input".into());
    }
    let context = rows.iter().map(|e| e.tokens.len()).max().unwrap();
    if !(2..=1024).contains(&vocabulary) || !(1..=4096).contains(&context) {
        return Err("invalid training dimensions".into());
    }
    let mut observations = vec![];
    let mut prefix_tokens = 0usize;
    for row in rows {
        if row.tokens.len() != row.targets.len()
            || row.tokens.len() != row.changes.len()
            || !row.prompt.is_empty()
            || !row.answer.is_empty()
            || row.field.is_some()
        {
            return Err("only aligned prefix-labeling records are supported".into());
        }
        for (position, &label) in row.targets.iter().enumerate() {
            if label >= 0 {
                prefix_tokens += position + 1;
                if observations.len() == 200_000 || prefix_tokens > 20_000_000 {
                    return Err("training prefix storage cap".into());
                }
                observations.push(Observation {
                    tokens: row.tokens[..=position]
                        .iter()
                        .map(|&t| usize::try_from(t))
                        .collect::<std::result::Result<_, _>>()?,
                    label: usize::try_from(label)?,
                });
            } else if label != -100 {
                return Err("invalid supervision mask".into());
            }
        }
    }
    let protocol: Value = serde_json::from_str(PROTOCOL)?;
    let out = binding::Request {
        discovery: complex::Request {
            input: Input {
                vocabulary,
                context,
                observations,
            },
            policy: serde_json::from_value(protocol["discovery"].clone())?,
        },
        training: serde_json::from_value(protocol["training"].clone())?,
    };
    out.validate()?;
    Ok(out)
}

#[cfg(test)]
mod tests;

/// Original success and the two99% goals are deliberately separate.
pub fn goals(metric: &Value) -> Value {
    let raw = metric["token_accuracy"].as_f64().unwrap();
    let exact = metric["sequence_accuracy"].as_f64().unwrap();
    json!({"original_success":exact >= 0.95,"raw99":raw >= 0.99,
        "exact99":exact >= 0.99,"both99":raw >= 0.99 && exact >= 0.99})
}

/// Reconstruct feature location with the scalar interpreter and multiply the
/// saved full complex head in class-major order, independently of logits_state.
fn independent(model: &Model, prefix: &[usize]) -> Vec<f64> {
    let Some(block) = model
        .blocks
        .iter()
        .find(|b| Some(&b.query) == prefix.last())
    else {
        return vec![0.; model.vocabulary];
    };
    let library = features::library(model.vocabulary, model.context);
    let value = block
        .decoder
        .candidate
        .original(&library, model.vocabulary, prefix);
    let state = block
        .decoder
        .values
        .iter()
        .position(|&v| v == value)
        .unwrap_or(block.states() - 1);
    (0..model.vocabulary)
        .map(|class| {
            (0..block.states())
                .map(|f| {
                    let row = (block.first_row + 2 * f) * model.vocabulary;
                    (complex::C::new(
                        model.coefficients[row + class],
                        model.coefficients[row + model.vocabulary + class],
                    ) * block.code[state * block.states() + f])
                        .re
                })
                .sum()
        })
        .collect()
}

fn chosen(z: &[f64]) -> usize {
    (1..z.len()).fold(0, |best, c| if z[c] > z[best] { c } else { best })
}

pub fn evaluate(model: &Model, splits: &BTreeMap<String, Vec<data::Example>>) -> Result<Value> {
    model.validate()?;
    let targets = splits
        .values()
        .flatten()
        .map(|e| e.targets.iter().filter(|&&y| y >= 0).count())
        .sum::<usize>();
    let states = model.blocks.iter().map(|b| b.states()).max().unwrap();
    // Metrics plus an independent complete-product replay, feature passes and
    // serialization. This bound includes all training and longer probe splits.
    let work = targets as u64
        * (64 * states as u64 * model.vocabulary as u64 + 16 * (4096 + model.vocabulary) as u64);
    if work > 2_000_000_000 {
        return Err("complete split scoring exceeds2B scalar-work forecast".into());
    }
    let mut out = BTreeMap::new();
    for (name, rows) in splits {
        let metric = metrics::evaluate(model, rows)?;
        let (mut known, mut known_errors, mut unknown_errors) = (0, 0, 0);
        let mut max_difference = 0.0f64;
        for row in rows {
            for (position, &label) in row.targets.iter().enumerate().filter(|(_, y)| **y >= 0) {
                let prefix: Vec<_> = row.tokens[..=position]
                    .iter()
                    .map(|&t| t as usize)
                    .collect();
                let actual = model.logits_tokens(&prefix);
                let checked = independent(model, &prefix);
                for (a, b) in actual.iter().zip(&checked) {
                    max_difference = max_difference.max((a - b).abs());
                }
                if max_difference > 1e-10 || chosen(&actual) != chosen(&checked) {
                    return Err("independent complete complex prediction differs".into());
                }
                let error = usize::from(chosen(&actual) != label as usize);
                if model.known(&prefix) {
                    known += 1;
                    known_errors += error;
                } else {
                    unknown_errors += error;
                }
            }
        }
        out.insert(
            name,
            json!({"goals":goals(&metric),"metrics":metric,
            "known_targets":known,"known_errors":known_errors,"unknown_errors":unknown_errors,
            "independent_complex_max_difference":max_difference}),
        );
    }
    Ok(json!({"splits":out,"scalar_work_forecast":work}))
}

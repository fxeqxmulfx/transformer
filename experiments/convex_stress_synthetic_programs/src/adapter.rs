//! Original masks and causal shifts, confined to the data-facing adapter.
//! IDs enumerate supervised positions in original example/position order.
use crate::{observations::Observation, *};
use dataset::data::Example;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub vocabulary: usize,
    pub observations: Vec<Observation>,
}

pub fn validate(e: &Example, vocabulary: usize) -> Result<()> {
    if e.tokens.is_empty()
        || e.tokens.len() > 4096
        || e.field.is_some()
        || e.tokens.len() != e.targets.len()
        || e.tokens.len() != e.changes.len()
        || e.tokens.iter().any(|&t| t < 0 || t as usize >= vocabulary)
        || e.targets
            .iter()
            .any(|&y| y != -100 && (y < 0 || y as usize >= vocabulary))
    {
        return Err("invalid original sequence layout or class".into());
    }
    let mut previous = None;
    for (&y, &changed) in e.targets.iter().zip(&e.changes) {
        if changed != (y >= 0 && previous != Some(y)) {
            return Err("original transition mask differs".into());
        }
        if y >= 0 {
            previous = Some(y);
        }
    }
    if e.prompt.is_empty() {
        if !e.answer.is_empty() || e.limit != 0 {
            return Err("mixed prefix/generation layout".into());
        }
    } else if e.answer.is_empty()
        || e.prompt.last() != Some(&18)
        || e.answer.last() != Some(&17)
        || e.answer[..e.answer.len() - 1].contains(&17)
        || e.limit < e.answer.len()
        || e.prompt.len() + e.limit > 4096
        || e.tokens
            != e.prompt
                .iter()
                .chain(&e.answer[..e.answer.len() - 1])
                .copied()
                .collect::<Vec<_>>()
        || e.targets
            != std::iter::repeat_n(-100, e.prompt.len() - 1)
                .chain(e.answer.iter().copied())
                .collect::<Vec<_>>()
    {
        return Err("invalid original shifted answer or generation limit".into());
    }
    Ok(())
}

pub fn input(vocabulary: usize, examples: &[Example]) -> Result<Input> {
    if examples.is_empty() || examples.len() > 20000 {
        return Err("example count refused".into());
    }
    let (mut count, mut positions) = (0usize, 0usize);
    for e in examples {
        validate(e, vocabulary)?;
        for (i, &y) in e.targets.iter().enumerate() {
            if y >= 0 {
                count += 1;
                positions += i + 1;
            }
        }
    }
    if !(2..=200000).contains(&count) || positions > 20_000_000 {
        return Err("observed storage forecast refused".into());
    }
    let mut observations = Vec::with_capacity(count);
    for e in examples {
        for (i, &y) in e.targets.iter().enumerate() {
            if y >= 0 {
                observations.push(Observation {
                    id: observations.len(),
                    tokens: e.tokens[..=i].iter().map(|&t| t as usize).collect(),
                    label: y as usize,
                });
            }
        }
    }
    observations::validate(&observations, vocabulary)?;
    Ok(Input {
        vocabulary,
        observations,
    })
}

/// Generates only the original training split. No validation/test pool or
/// selected target-program metadata is supplied to discovery or fitting.
pub fn training_only(case: &str) -> Result<Input> {
    let (spec, _) = from_data(dataset::dataset::specification(case))?;
    let count = if spec["task"] == "random_lm" { 8 } else { 128 };
    let examples = from_data(dataset::dataset::generate(&spec, "train", count))?;
    input(from_data(dataset::spec::vocabulary(&spec))?, &examples)
}

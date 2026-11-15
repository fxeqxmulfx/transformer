//! Causal original records, borrowed rather than copying every prefix. Ported
//! masks/shift validation from synthetic_programs/adapter.rs at16aafe00.
use crate::*;
use basis::data::Example;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub struct Observation<'a> {
    pub id: usize,
    pub row: usize,
    pub position: usize,
    pub tokens: &'a [i64],
    pub label: usize,
}
pub fn observations(rows: &[Example]) -> impl Iterator<Item = Observation<'_>> {
    rows.iter()
        .enumerate()
        .flat_map(|(row, e)| {
            e.targets
                .iter()
                .enumerate()
                .filter(|(_, y)| **y >= 0)
                .map(move |(position, &y)| (row, position, &e.tokens[..=position], y as usize))
        })
        .enumerate()
        .map(|(id, (row, position, tokens, label))| Observation {
            id,
            row,
            position,
            tokens,
            label,
        })
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub vocabulary: usize,
    pub records: usize,
    pub examples: usize,
    pub prefix_positions: u64,
    pub maximum_length: usize,
    pub alphabet: Vec<usize>,
    pub observed_labels: Vec<usize>,
    pub training_tensor_sha256: String,
}
pub fn inspect(vocabulary: usize, rows: &[Example]) -> Result<Input> {
    if !(2..=1024).contains(&vocabulary) || rows.is_empty() || rows.len() > 20000 {
        return Err("original training input dimensions refused".into());
    }
    let mut positions = 0;
    for e in rows {
        previous::adapter::validate(e, vocabulary)?;
        positions += e.tokens.len();
    }
    if positions > 2_000_000 {
        return Err("original training position bound exceeded".into());
    }
    let (mut count, mut prefix_positions, mut maximum_length) = (0, 0u64, 0);
    let (mut alphabet, mut labels) = (BTreeSet::new(), BTreeSet::new());
    for r in observations(rows) {
        count += 1;
        prefix_positions += r.tokens.len() as u64;
        maximum_length = maximum_length.max(r.tokens.len());
        alphabet.extend(r.tokens.iter().map(|&x| x as usize));
        labels.insert(r.label);
    }
    if !(2..=2_000_000).contains(&count) {
        return Err("original supervised count refused".into());
    }
    Ok(Input {
        vocabulary,
        records: count,
        examples: rows.len(),
        prefix_positions,
        maximum_length,
        alphabet: alphabet.into_iter().collect(),
        observed_labels: labels.into_iter().collect(),
        training_tensor_sha256: basis::wire::tensor_hash(rows),
    })
}

/// Data generation only. The fitting API receives no case name or specification,
/// and only the corrupted observed pool is returned for the noisy case.
pub fn training_only(case: &str) -> Result<(usize, Vec<Example>)> {
    let s = from_data(basis::dataset::specification(case))?;
    let noisy = s["task"] == "parity";
    let count = if noisy { 512 } else { 20000 };
    let clean = from_data(basis::generator::generate(
        &s,
        "train",
        count,
        &BTreeSet::new(),
        noisy,
    ))?;
    let rows = if noisy {
        from_data(basis::noise::apply(&clean, 0.2, 2))?.0
    } else {
        clean
    };
    let vocabulary = if s["task"] == "mqar" {
        36 + 2 * from_data(basis::generator::number(&s, "symbols"))?
    } else if noisy {
        68
    } else {
        36
    };
    Ok((vocabulary, rows))
}

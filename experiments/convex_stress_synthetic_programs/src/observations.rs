//! The sole supervised input to discovery and convex fitting. Original IDs
//! distinguish repeated prefixes. Dataset metadata and target-program
//! descriptions have no representation in this interface.
use crate::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Observation {
    pub id: usize,
    pub tokens: Vec<usize>,
    pub label: usize,
}

pub fn validate(rows: &[Observation], vocabulary: usize) -> Result<()> {
    if !(2..=200000).contains(&rows.len()) || !(2..=1024).contains(&vocabulary) {
        return Err("observed dimensions refused".into());
    }
    let mut ids = BTreeSet::new();
    let mut positions = 0usize;
    for r in rows {
        if r.tokens.is_empty()
            || r.tokens.len() > 4096
            || r.tokens.iter().any(|&x| x >= vocabulary)
            || r.label >= vocabulary
            || !ids.insert(r.id)
        {
            return Err("invalid observed prefix, label or original record ID".into());
        }
        positions = positions
            .checked_add(r.tokens.len())
            .ok_or("observed size overflow")?;
    }
    if positions > 20_000_000 {
        return Err("observed prefix-storage limit refused".into());
    }
    Ok(())
}

pub fn coordinates(vocabulary: usize) -> Result<Vec<f64>> {
    if !(2..=1024).contains(&vocabulary) {
        return Err("invalid class geometry size".into());
    }
    // Explicit ordinal geometry. Relabeling must transport the coordinates.
    let middle = (vocabulary - 1) as f64 / 2.;
    Ok((0..vocabulary)
        .map(|c| (c as f64 - middle) / middle)
        .collect())
}

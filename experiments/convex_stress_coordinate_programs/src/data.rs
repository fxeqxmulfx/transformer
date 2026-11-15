//! Separate input-only features from the observed label and original record ID.
use crate::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Record {
    pub id: usize,
    pub features: Vec<Features>,
    pub label: usize,
}

pub fn coordinates(values: &[f64]) -> Result<()> {
    if !(2..=1024).contains(&values.len()) || values.iter().any(|e| !e.is_finite() || e.abs() > 1.)
    {
        return Err("invalid supplied class geometry".into());
    }
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    if ordered.windows(2).any(|w| w[1] - w[0] < 1e-8) {
        return Err("class coordinates are not numerically separated".into());
    }
    Ok(())
}

pub fn features(rows: &[Features], programs: usize) -> Result<()> {
    if rows.len() != programs
        || rows
            .iter()
            .flatten()
            .any(|x| !x.is_finite() || x.abs() > 4096.)
    {
        return Err("invalid input-only program features".into());
    }
    Ok(())
}

/// Conservative bound includes records, complete coefficients, domain rows,
/// optimizer buffers and serialization copies before their allocation.
pub fn forecast(n: usize, p: usize, v: usize) -> Result<u64> {
    if !(2..=MAX_RECORDS).contains(&n)
        || !(1..=MAX_PROGRAMS).contains(&p)
        || !(2..=1024).contains(&v)
    {
        return Err("coordinate-program dimensions refused".into());
    }
    let (n, p, v) = (n as u128, p as u128, v as u128);
    let bytes = 256 * 1024 * 1024 + 64 * (n * (128 + 88 * p) + 256 * p * v + 4096 * p);
    if bytes > MAX_BYTES {
        return Err("coordinate-program buffer forecast exceeds56GiB".into());
    }
    Ok(bytes as u64)
}

pub fn records(rows: &[Record], coords: &[f64], programs: usize) -> Result<u64> {
    let bytes = forecast(rows.len(), programs, coords.len())?;
    coordinates(coords)?;
    let mut ids = BTreeSet::new();
    for r in rows {
        features(&r.features, programs)?;
        if r.label >= coords.len() || !ids.insert(r.id) {
            return Err("invalid label or repeated record ID".into());
        }
    }
    Ok(bytes)
}

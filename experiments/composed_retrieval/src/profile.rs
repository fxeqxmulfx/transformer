//! The original ten-case comparison changes only the shape of the signed
//! positional law. Reuse all verified data from the parent 64-case cohort.
use crate::{
    Result,
    data::{self, PROFILE_PROTOCOL, Protocol},
    radii_cohort,
};
use baseline::hash;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

#[derive(Deserialize)]
struct Definition {
    name: String,
    training_protocol_sha256: String,
    cases: BTreeSet<String>,
    initial_adjacent_level: f64,
    initial_decay: f64,
    minimum_decay: f64,
    learning_rate: String,
}
pub fn configure(p: &mut Protocol) -> Result<()> {
    let d: Definition = serde_json::from_str(PROFILE_PROTOCOL)?;
    if d.training_protocol_sha256 != hash(data::RADII_PROTOCOL.as_bytes())
        || !p.learned_product_radii
        || p.cases.len() != 64
        || p.coordinates_per_table.is_some()
        || d.initial_adjacent_level != 0.
        || d.initial_decay != 0.
        || d.minimum_decay != 0.
        || d.learning_rate != "routing_lr"
        || d.cases.len() != 10
        || !d.cases.iter().all(|id| p.cases.contains_key(id))
    {
        return Err("signed profile or original source roster differs".into());
    }
    p.cases.retain(|id, _| d.cases.contains(id));
    p.name = d.name;
    p.signed_profile = true;
    Ok(())
}

pub fn inventory(p: &Protocol, inputs: &Path) -> Result<Value> {
    let parent = data::parent_protocol()?;
    let source = radii_cohort::preflight(&parent, inputs)?;
    let mut entries = BTreeMap::new();
    let mut one = 0u64;
    for (id, entry) in source["cases"].as_object().ok_or("original parent cases")? {
        if let Some(spec) = p.cases.get(id) {
            if parent.cases.get(id) != Some(spec) {
                return Err("unmatched source specification".into());
            }
            let (task, pins) = data::original(p, id)?;
            if entry["split_fingerprints"] != pins {
                return Err("mismatched original splits".into());
            }
            let queries: usize = task
                .splits
                .values()
                .flatten()
                .map(|row| row.targets.iter().filter(|&&y| y >= 0).count() + row.limit)
                .sum();
            one += queries as u64 * 8 * task.vocab as u64;
        } else if !entry["dimensions"].is_null() {
            continue;
        }
        entries.insert(id, entry);
    }
    let fits = p.seeds.len() * p.modes.len() * p.cases.len() * 2;
    if entries.len() != 16 || fits != 120 {
        return Err("incomplete pilot or missing-source accounting".into());
    }
    Ok(
        json!({"protocol_sha256":hash(PROFILE_PROTOCOL.as_bytes()),"cases":entries,"fits":fits,
        "raw_bytes_upper_bound":one*4*p.seeds.len() as u64*p.modes.len() as u64}),
    )
}

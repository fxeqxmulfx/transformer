//! Matched ten-case studies of signed distances and predecessor matching.
//! Each study reuses the same original examples and complete CM117 training.
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
        || d.initial_adjacent_level != 0.
        || d.initial_decay != 0.
        || d.minimum_decay != 0.
        || d.learning_rate != "routing_lr"
    {
        return Err("signed profile or original source roster differs".into());
    }
    select(p, &d.cases)?;
    p.name = d.name;
    p.signed_profile = true;
    Ok(())
}

fn select(p: &mut Protocol, cases: &BTreeSet<String>) -> Result<()> {
    if !p.learned_product_radii
        || p.signed_profile
        || p.predecessor_match
        || p.cases.len() != 64
        || p.coordinates_per_table.is_some()
        || cases.len() != 10
        || !cases.iter().all(|id| p.cases.contains_key(id))
    {
        return Err("unexpected model, training budget or pilot roster".into());
    }
    p.cases.retain(|id, _| cases.contains(id));
    Ok(())
}

pub fn configure_predecessor(p: &mut Protocol) -> Result<()> {
    let d: Value = serde_json::from_str(data::PREDECESSOR_PROTOCOL)?;
    if d["training_protocol_sha256"] != hash(data::RADII_PROTOCOL.as_bytes())
        || d["roster_protocol_sha256"] != hash(PROFILE_PROTOCOL.as_bytes())
        || d["initial_coefficient"] != 0
        || d["learning_rate"] != "routing_lr"
    {
        return Err("changed contextual matching or pilot source".into());
    }
    let roster: Definition = serde_json::from_str(PROFILE_PROTOCOL)?;
    select(p, &roster.cases)?;
    p.name = d["name"]
        .as_str()
        .ok_or("missing predecessor study name")?
        .into();
    p.predecessor_match = true;
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
        json!({"protocol_sha256":hash(data::PROTOCOL.as_bytes()),"cases":entries,"fits":fits,
        "raw_bytes_upper_bound":one*4*p.seeds.len() as u64*p.modes.len() as u64}),
    )
}

//! The 64 distinct original cases share one learned-radius training rule.
//! Merge only identical source specifications on the three repeated entries.
use crate::{
    Result, arithmetic,
    data::{self, Protocol},
    prefix,
};
use baseline::hash;
use serde_json::Value;
use std::{collections::btree_map::Entry, path::Path};

fn sources() -> [(&'static str, bool); 2] {
    [
        (data::PREFIX_PROTOCOL, false),
        (data::REWRITE_PROTOCOL, true),
    ]
}
pub fn configure(p: &mut Protocol) -> Result<()> {
    let d: Value = serde_json::from_str(data::RADII_PROTOCOL)?;
    if d["parent_training_sha256"] != hash(data::OFFSET_PROTOCOL.as_bytes())
        || d["prefix_source_sha256"] != hash(data::PREFIX_PROTOCOL.as_bytes())
        || d["rewrite_source_sha256"] != hash(data::REWRITE_PROTOCOL.as_bytes())
        || d["initial_radius"] != 1
        || d["minimum_radius"] != 0
        || d["maximum_radius"] != 1
        || d["radius_learning_rate"] != p.gate_lr
        || p.cases.len() != 51
        || p.coordinates_per_table.is_some()
    {
        return Err("changed training, radius domain, or source reference".into());
    }
    let mut overlaps = 0;
    for (text, rewrite) in sources() {
        let mut group = p.clone();
        prefix::configure_from(&mut group, text, rewrite)?;
        for (id, s) in group.cases {
            match p.cases.entry(id) {
                Entry::Occupied(e) if e.get() == &s => {
                    overlaps += 1;
                }
                Entry::Occupied(_) => return Err("same case has different source manifests".into()),
                Entry::Vacant(e) => {
                    e.insert(s);
                }
            }
        }
    }
    if p.cases.len() != 64
        || d["measured_cases"] != 64
        || overlaps != 3
        || d["unavailable_arithmetic_sources"] != 6
    {
        return Err("incomplete combined original roster".into());
    }
    p.learned_product_radii = true;
    p.name = d["name"]
        .as_str()
        .ok_or("missing learned-radius protocol name")?
        .into();
    Ok(())
}

pub fn preflight(p: &Protocol, inputs: &Path) -> Result<Value> {
    let all = arithmetic::preflight(p, inputs)?;
    for (text, rewrite) in sources() {
        let mut group = p.clone();
        prefix::configure_from(&mut group, text, rewrite)?;
        let pins = prefix::inventory_from(&group, inputs, text, rewrite)?;
        for (id, entry) in pins["cases"]
            .as_object()
            .ok_or("synthetic source entries")?
        {
            if entry["split_fingerprints"] != all["cases"][id]["split_fingerprints"] {
                return Err("source inventories disagree".into());
            }
        }
    }
    if all["cases"].as_object().is_none_or(|x| x.len() != 70)
        || all["fits"] != 768
        || !all["raw_bytes_upper_bound"].is_u64()
    {
        return Err("incomplete raw storage or fit forecast".into());
    }
    Ok(all)
}

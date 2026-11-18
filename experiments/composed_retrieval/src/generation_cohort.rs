//! Rebuild the original generation pools and score their original selection.
//! Only source records, batch size and object dimensions reach fitting.
use crate::{
    Result,
    data::{CaseSpec, GENERATION_PROTOCOL, Protocol, RADII_PROTOCOL},
    generation, prefix,
};
use baseline::{data::Task, hash};
use learned_retrieval::cohort::SUITE;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

pub const NAME: &str = "composed-retrieval-original-generation24-product-radii-v1";
pub const OVERLAP: &str = "suite-copy-repeat";
const BOOLEAN: [&str; 2] = ["suite-boolean-and-shift", "suite-boolean-and-random"];

#[derive(Deserialize)]
struct Definition {
    name: String,
    training_protocol_sha256: String,
    plan_sha256: String,
    frequency_sha256: String,
    parity_sha256: String,
    remaining_sha256: String,
    selection: String,
    metric: String,
    target: f64,
    batch: usize,
    cases: BTreeMap<String, prefix::Source>,
}

pub fn configure(p: &mut Protocol) -> Result<()> {
    let d: Definition = serde_json::from_str(GENERATION_PROTOCOL)?;
    let expected: BTreeSet<_> = frequency_data::CASES
        .into_iter()
        .chain(parity_data::CASES)
        .chain(
            synthetic_data::CASES
                .into_iter()
                .filter(|id| !id.starts_with("suite-crasp-")),
        )
        .chain(BOOLEAN)
        .chain([OVERLAP])
        .collect();
    if !p.learned_product_radii
        || p.cases.len() != 64
        || d.name != NAME
        || d.training_protocol_sha256 != hash(RADII_PROTOCOL.as_bytes())
        || d.frequency_sha256 != hash(frequency_data::PROTOCOL.as_bytes())
        || d.parity_sha256 != hash(parity_data::PROTOCOL.as_bytes())
        || d.remaining_sha256 != hash(synthetic_data::PROTOCOL.as_bytes())
        || expected.len() != 24
        || expected != d.cases.keys().map(String::as_str).collect()
    {
        return Err("generation roster or frozen training and source protocol differs".into());
    }
    let cases: BTreeMap<_, _> = d
        .cases
        .into_iter()
        .map(|(id, s)| {
            let spec = CaseSpec {
                vocabulary: s.vocabulary,
                context: s.context,
                batch: d.batch,
                train: if id == "random-lm-control" { 8 } else { 128 },
                selection: d.selection.clone(),
                metric: d.metric.clone(),
                target: d.target,
                metadata_sha256: Some(s.metadata_sha256),
                source_fingerprints: None,
            };
            (id, spec)
        })
        .collect();
    if p.cases.get(OVERLAP) != cases.get(OVERLAP)
        || cases.keys().filter(|id| p.cases.contains_key(*id)).count() != 1
    {
        return Err("expected exactly the original copy overlap".into());
    }
    p.cases = cases;
    p.name = d.name;
    Ok(())
}

/// The historical 35-case subset preserves all base split hashes and the
/// selected novel split. The full suite also scores novel validation probes.
/// Follow Memorization's original ordered prompt filter for each such probe.
fn boolean(id: &str) -> Result<(Task, Value)> {
    let suite: Value = serde_json::from_str(SUITE)?;
    let row = suite["tasks"]
        .as_array()
        .ok_or("Boolean source suite")?
        .iter()
        .find(|r| r["name"] == id)
        .ok_or("missing original Boolean row")?;
    let mut task = baseline::data::build(row)?;
    let members: BTreeSet<_> = task.splits["train_clean"]
        .iter()
        .map(|e| &e.prompt)
        .collect();
    let mut subsets = Vec::new();
    for (name, rows) in &task.splits {
        if !name.starts_with("validation/") || name.ends_with("/novel") {
            continue;
        }
        let novel: Vec<_> = rows
            .iter()
            .filter(|e| !members.contains(&e.prompt))
            .cloned()
            .collect();
        if !novel.is_empty() {
            subsets.push((format!("{name}/novel"), novel));
        }
    }
    task.splits.extend(subsets);
    Ok((task, row["data_fingerprints"].clone()))
}

pub fn original(id: &str, spec: &CaseSpec) -> Result<(Task, Value)> {
    if id == OVERLAP {
        return prefix::original(id, spec);
    }
    let (t, meta) = if frequency_data::CASES.contains(&id) {
        let d = frequency_data::dataset::build(id)?;
        let meta = frequency_data::dataset::metadata(&d);
        (d.task, meta)
    } else if parity_data::CASES.contains(&id) {
        let d = parity_data::dataset::build(id)?;
        let meta = parity_data::dataset::metadata(&d);
        (d.task, meta)
    } else if BOOLEAN.contains(&id) {
        boolean(id)?
    } else {
        let d = synthetic_data::dataset::build(id)?;
        let meta = synthetic_data::dataset::metadata(&d)?;
        (d.task, meta)
    };
    let mut context = 0;
    for row in t.splits.values().flatten() {
        generation::validate_shifted(row, t.vocab)?;
        context = context.max(row.prompt.len() + row.limit - 1);
    }
    if spec.metadata_sha256.as_deref() != Some(&hash(&serde_json::to_vec(&meta)?))
        || spec.context != context
        || spec.vocabulary != t.vocab
        || spec.selection != t.selection
        || spec.train != t.splits["train"].len()
    {
        return Err(format!(
            "original generation source or dimensions differ on {id} (context {context})"
        )
        .into());
    }
    Ok((t, meta))
}

pub fn preflight(p: &Protocol, inputs: &Path) -> Result<Value> {
    let d: Definition = serde_json::from_str(GENERATION_PROTOCOL)?;
    let mut out = prefix::source_inventory(
        p,
        inputs,
        GENERATION_PROTOCOL,
        &d.plan_sha256,
        &d.cases,
        true,
        original,
    )?;
    let id = "random-lm-control";
    let (control, _) = original(id, &p.cases[id])?;
    let mut bounds = BTreeMap::new();
    for (name, rows) in &control.splits {
        let first = rows.first().ok_or("empty control split")?;
        let mut frequency = BTreeMap::new();
        for row in rows {
            if row.prompt != first.prompt || row.limit != first.limit {
                return Err("random control exposes different prompts or generation bounds".into());
            }
            *frequency.entry(&row.answer).or_insert(0usize) += 1;
        }
        bounds.insert(name, json!([frequency.values().max(), rows.len()]));
    }
    if bounds.get(&control.selection) != Some(&json!([1, 64])) {
        return Err("random control selection ceiling changed".into());
    }
    out["random_lm_maximum_exact"] = json!(bounds);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boolean_validation_probes_retain_every_novel_row_in_original_order() -> Result<()> {
        let p = crate::data::protocol()?;
        for (id, splits, primary) in [(BOOLEAN[0], 20, 7), (BOOLEAN[1], 11, 8)] {
            let s = &p.cases[id];
            assert_eq!(s.target, 0.95);
            let (task, pins) = original(id, s)?;
            assert_eq!(task.splits.len(), splits);
            assert_eq!(task.splits[&task.selection].len(), primary);
            let members: BTreeSet<_> = task.splits["train_clean"]
                .iter()
                .map(|e| &e.prompt)
                .collect();
            for name in pins
                .as_object()
                .ok_or("original source pools")?
                .keys()
                .filter(|name| name.starts_with("validation"))
            {
                let expected: Vec<_> = task.splits[name]
                    .iter()
                    .filter(|e| !members.contains(&e.prompt))
                    .map(|e| (&e.tokens, &e.targets, &e.changes))
                    .collect();
                let novel = format!("{name}/novel");
                let actual: Vec<_> = task.splits[&novel]
                    .iter()
                    .map(|e| (&e.tokens, &e.targets, &e.changes))
                    .collect();
                assert_eq!(actual, expected);
            }
        }
        Ok(())
    }
}

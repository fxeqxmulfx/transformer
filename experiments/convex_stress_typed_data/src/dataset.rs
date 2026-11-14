//! Original seed1 typed-bracket splits, including clean/novel and32/64 probes.
//! The enclosing inventory labels their fingerprints as source-derived only.
use crate::*;
pub use sequences::dataset::Dataset;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

fn integer(spec: &Value, key: &str) -> Result<usize> {
    usize::try_from(
        spec[key]
            .as_u64()
            .ok_or("missing nonnegative data parameter")?,
    )
    .map_err(Into::into)
}
pub fn generate(spec: &Value, split: &str, count: usize) -> Result<Vec<data::Example>> {
    let length = integer(spec, "length")?;
    let minimum = if spec["min_length"].is_null() {
        length
    } else {
        integer(spec, "min_length")?
    };
    let types = integer(spec, "bracket_types")?;
    let maximum = integer(spec, "max_balance")?;
    if spec["task"] != "dyck2"
        || spec["neutral_fraction"] != 0.25
        || !spec["max_neutral_gap"].is_null()
        || !(2..=8).contains(&types)
        || !(1..=4096).contains(&maximum)
        || !["train", "validation", "test"].contains(&split)
        || count == 0
        || count > 20000
        || minimum < 7
        || minimum > length
        || length > 4096
        || count * length > 2_000_000
    {
        return Err("typed-bracket generator parameters refused".into());
    }
    let mut rng = wire::rng(spec, split);
    let mut out = Vec::with_capacity(count + 1);
    while out.len() < count {
        let n = minimum + rng.below((length - minimum + 1) as u32) as usize;
        let mut pair = generator::pair(types, n, maximum as i64, &mut rng);
        if rng.random() < 0.5 {
            pair.reverse();
        }
        out.extend(pair);
    }
    out.truncate(count);
    Ok(out)
}
pub fn build(case: &str) -> Result<Dataset> {
    if !CASES.contains(&case) {
        return Err("case outside frozen typed cohort".into());
    }
    let protocol: Value = serde_json::from_str(PROTOCOL)?;
    let mut spec = protocol["definitions"][case].clone();
    let lengths = spec
        .as_object_mut()
        .ok_or("missing typed definition")?
        .remove("longer_lengths")
        .ok_or("missing probes")?;
    let vocabulary = 36 + 2 * (integer(&spec, "bracket_types")? - 1);
    let mut splits: BTreeMap<String, Vec<data::Example>> = BTreeMap::new();
    let mut fingerprints = BTreeMap::new();
    let mut add = |name: &str, split: &str, count: usize, spec: &Value| -> Result<()> {
        let rows = generate(spec, split, count)?;
        fingerprints.insert(name.into(), wire::split_hash(spec, split, &rows));
        splits.insert(name.into(), rows);
        Ok(())
    };
    for (split, count) in [("train", 128), ("validation", 32), ("test", 64)] {
        add(split, split, count, &spec)?;
    }
    add("train_clean", "train", 128, &spec)?;
    for length in lengths.as_array().ok_or("invalid probes")? {
        let mut probe = spec.clone();
        probe["length"] = length.clone();
        probe["min_length"] = Value::Null;
        for (split, count) in [("validation", 32), ("test", 64)] {
            add(&format!("{split}/length-{length}"), split, count, &probe)?;
        }
    }
    let train: BTreeSet<_> = splits["train"].iter().map(|r| r.tokens.clone()).collect();
    let names: Vec<String> = splits
        .keys()
        .filter(|name| name.starts_with("validation"))
        .cloned()
        .collect();
    let mut empty = vec![];
    for name in names {
        let novel: Vec<_> = splits[&name]
            .iter()
            .filter(|r| !train.contains(&r.tokens))
            .cloned()
            .collect();
        let key = format!("{name}/novel");
        if novel.is_empty() {
            empty.push(key);
        } else {
            splits.insert(key, novel);
        }
    }
    let selection = if splits.contains_key("validation/novel") {
        "validation/novel"
    } else {
        "validation"
    };
    let tensor_fingerprints = splits
        .iter()
        .map(|(name, rows)| (name.clone(), wire::tensor_hash(rows)))
        .collect();
    Ok(Dataset {
        name: case.into(),
        task: data::Task {
            vocab: vocabulary,
            selection: selection.into(),
            splits,
        },
        original_fingerprints: fingerprints,
        tensor_fingerprints,
        empty_novel_splits: empty,
    })
}

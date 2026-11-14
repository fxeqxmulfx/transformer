//! Source-derived first-cohort splits, with original/novel selection retained.
//! Only this adapter sees task identities. Predictors receive prefixes and labels.
use crate::{CATALOG, Result, data, prefix, retrieval, wire};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

pub struct Dataset {
    pub name: String,
    pub task: data::Task,
    pub original_fingerprints: BTreeMap<String, String>,
    pub tensor_fingerprints: BTreeMap<String, String>,
    pub empty_novel_splits: Vec<String>,
}

fn integer(spec: &Value, name: &str) -> Result<usize> {
    usize::try_from(
        spec[name]
            .as_u64()
            .ok_or("missing nonnegative generator parameter")?,
    )
    .map_err(Into::into)
}

pub fn generate(spec: &Value, split: &str, count: usize) -> Result<Vec<data::Example>> {
    let length = integer(spec, "length")?;
    let minimum = match spec.get("min_length").ok_or("missing minimum length")? {
        Value::Null => length,
        value => usize::try_from(value.as_u64().ok_or("invalid minimum length")?)?,
    };
    let task = spec["task"].as_str().ok_or("missing generator family")?;
    if !["train", "validation", "test"].contains(&split)
        || count == 0
        || count > 20_000
        || minimum < 1
        || minimum > length
        || length > 4096
        || count * length > 2_000_000
    {
        return Err("generator size or split refused".into());
    }
    match task {
        "mqar" | "lookup" => {
            let symbols = integer(spec, "symbols")?;
            let pairs = integer(spec, "pairs")?;
            let queries = integer(spec, "queries")?;
            let gap = integer(spec, "query_gap")?;
            if symbols > 1024
                || pairs == 0
                || pairs > symbols
                || queries == 0
                || queries > pairs
                || gap > 4096
            {
                return Err("retrieval parameters refused".into());
            }
            if task == "mqar" {
                let alpha = spec["alpha"].as_f64().ok_or("missing MQAR alpha")?;
                if minimum < 1 + 2 * pairs + queries + gap
                    || !alpha.is_finite()
                    || alpha < 0.
                    || spec.get("overwrites").is_some()
                {
                    return Err("unsupported MQAR distribution".into());
                }
            } else {
                let hops = integer(spec, "hops")?;
                if hops == 0 || hops >= pairs || minimum < 2 + 4 * pairs + 2 * queries + gap {
                    return Err("lookup size or path refused".into());
                }
            }
        }
        "dyck" | "blocks" => {
            if spec["neutral_fraction"] != 0.25 || !spec["max_neutral_gap"].is_null() {
                return Err("unsupported prefix neutral distribution".into());
            }
            let blocks = if task == "blocks" {
                integer(spec, "blocks")?
            } else {
                0
            };
            if (task == "blocks" && (blocks == 0 || blocks > 1024))
                || (task == "dyck" && !(1..=4096).contains(&integer(spec, "max_balance")?))
            {
                return Err("prefix parameter refused".into());
            }
            for length in minimum..=length {
                let mut active = ((length - 1) as f64 * 0.75).round_ties_even() as usize;
                if task == "dyck" {
                    active -= active % 2;
                }
                let needed = if task == "dyck" {
                    6
                } else {
                    blocks + if blocks > 1 { 2 } else { 1 }
                };
                if active < needed {
                    return Err("prefix too short".into());
                }
            }
        }
        _ => return Err("unsupported native sequence generator".into()),
    }
    let mut rng = wire::rng(spec, split);
    let mut out = Vec::with_capacity(count + 1);
    while out.len() < count {
        let length = minimum + rng.below((length - minimum + 1) as u32) as usize;
        let mut pair = match task {
            "mqar" => (0..2)
                .map(|_| retrieval::mqar(spec, &mut rng, length))
                .collect(),
            "lookup" => retrieval::lookup(spec, &mut rng, length),
            _ => prefix::pair(spec, &mut rng, length),
        };
        if rng.random() < 0.5 {
            pair.reverse();
        }
        out.extend(pair);
    }
    out.truncate(count);
    Ok(out)
}

pub fn build(name: &str) -> Result<Dataset> {
    let catalog: Value = serde_json::from_str(CATALOG)?;
    if !catalog["first_cohort"]
        .as_array()
        .unwrap()
        .iter()
        .any(|id| id == name)
    {
        return Err("task outside frozen first cohort".into());
    }
    let mut spec = catalog["first_cohort_definitions"][name].clone();
    let lengths = spec
        .as_object_mut()
        .unwrap()
        .remove("longer_lengths")
        .unwrap();
    // The original MQAR controls omit a zero overwrite count from RNG identity.
    spec.as_object_mut().unwrap().remove("overwrites");
    let vocabulary = match spec["task"].as_str().unwrap() {
        "mqar" => 36 + 2 * integer(&spec, "symbols")?,
        "lookup" => 36 + integer(&spec, "symbols")?,
        _ => 36,
    };
    let mut splits = BTreeMap::new();
    let mut fingerprints = BTreeMap::new();
    let mut add = |name: &str, split: &str, count, current: &Value| -> Result<()> {
        let rows = generate(current, split, count)?;
        fingerprints.insert(name.to_string(), wire::split_hash(current, split, &rows));
        splits.insert(name.to_string(), rows);
        Ok(())
    };
    for (split, count) in [("train", 128), ("validation", 32), ("test", 64)] {
        add(split, split, count, &spec)?;
    }
    add("train_clean", "train", 128, &spec)?;
    for length in lengths.as_array().unwrap() {
        let mut current = spec.clone();
        current["length"] = length.clone();
        current["min_length"] = Value::Null;
        for (split, count) in [("validation", 32), ("test", 64)] {
            add(&format!("{split}/length-{length}"), split, count, &current)?;
        }
    }
    let train: BTreeSet<_> = splits["train"].iter().map(|e| e.tokens.clone()).collect();
    let names: Vec<_> = splits
        .keys()
        .filter(|s| s.starts_with("validation"))
        .cloned()
        .collect();
    let mut empty = vec![];
    for name in names {
        let novel: Vec<_> = splits[&name]
            .iter()
            .filter(|e| !train.contains(&e.tokens))
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
        name: name.into(),
        task: data::Task {
            splits,
            vocab: vocabulary,
            selection: selection.into(),
        },
        original_fingerprints: fingerprints,
        tensor_fingerprints,
        empty_novel_splits: empty,
    })
}

impl Dataset {
    pub fn metadata(&self) -> Value {
        let counts:BTreeMap<_,_>=self.task.splits.iter().map(|(name,rows)|
            (name.clone(),json!({"examples":rows.len(),"targets":rows.iter().map(|e|e.targets.iter().filter(|&&y|y>=0).count()).sum::<usize>(),
                "max_context":rows.iter().map(|e|e.tokens.len()).max().unwrap_or(0)}))).collect();
        json!({"case":self.name,"vocabulary":self.task.vocab,"selection":self.task.selection,
            "source_derived_split_fingerprints":self.original_fingerprints,
            "native_tensor_fingerprints":self.tensor_fingerprints,"splits":counts,
            "empty_novel_splits":self.empty_novel_splits,
            "historical_case_fingerprints_available":false,"new_model_evaluations":0})
    }
}

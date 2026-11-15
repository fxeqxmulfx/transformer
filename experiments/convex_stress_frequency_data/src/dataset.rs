//! Original seed1 frequency pools and their32/64-symbol probes.
//! Novelty keys are raw prompts, never teacher-forced answers.
use crate::*;
pub use sequences::dataset::Dataset;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

fn integer(spec: &Value, key: &str) -> Result<usize> {
    usize::try_from(spec[key].as_u64().ok_or("missing data integer")?).map_err(Into::into)
}
pub fn generate(spec: &Value, split: &str, count: usize) -> Result<Vec<data::Example>> {
    let task = spec["task"].as_str().ok_or("missing task")?;
    let length = integer(spec, "length")?;
    let minimum = if spec["min_length"].is_null() {
        length
    } else {
        integer(spec, "min_length")?
    };
    let symbols = integer(spec, "symbols")?;
    let scratchpad = spec["scratchpad"].as_str().unwrap_or("none");
    let numbers = uses_numbers(spec);
    let number_limit = if numbers {
        integer(spec, "number_limit")?
    } else {
        4096
    };
    let bos = if task == "histogram" {
        spec["histogram_bos"]
            .as_bool()
            .ok_or("missing BOS policy")?
    } else {
        true
    };
    if !["histogram", "histogram2", "mode", "most_freq"].contains(&task)
        || (task == "mode" && !["none", "counts", "itemized"].contains(&scratchpad))
        || !["train", "validation", "test"].contains(&split)
        || !(2..=1024).contains(&symbols)
        || minimum == 0
        || (task == "mode" && minimum < 3)
        || minimum > length
        || length > 1024
        || !(1..=4096).contains(&number_limit)
        || length > number_limit
        || count == 0
        || count > 20000
        || count * (4 * length + 3) > 2_000_000
    {
        return Err("frequency generator parameters refused".into());
    }
    let config = generator::Config {
        task,
        symbols,
        bos,
        scratchpad,
        numbers,
    };
    let mut rng = wire::rng(&identity(spec), split);
    let mut rows = Vec::with_capacity(count + 1);
    while rows.len() < count {
        let size = minimum + rng.below((length - minimum + 1) as u32) as usize;
        let mut pair: Vec<_> = (0..2)
            .map(|_| generator::sample(&config, size, &mut rng))
            .collect();
        if rng.random() < 0.5 {
            pair.reverse();
        }
        rows.extend(pair);
    }
    rows.truncate(count);
    Ok(rows)
}

pub fn build(case: &str) -> Result<Dataset> {
    if !CASES.contains(&case) {
        return Err("case outside frozen frequency cohort".into());
    }
    let p: Value = serde_json::from_str(PROTOCOL)?;
    let mut spec = p["definitions"][case].clone();
    let lengths = spec
        .as_object_mut()
        .ok_or("missing case definition")?
        .remove("longer_lengths")
        .ok_or("missing longer probes")?;
    let vocabulary = 36
        + integer(&spec, "symbols")?
        + if uses_numbers(&spec) {
            integer(&spec, "number_limit")? + 1
        } else {
            0
        };
    let mut splits: BTreeMap<String, Vec<data::Example>> = BTreeMap::new();
    let mut fingerprints = BTreeMap::new();
    let mut add = |name: &str, split: &str, count: usize, current: &Value| -> Result<()> {
        let rows = generate(current, split, count)?;
        fingerprints.insert(
            name.into(),
            wire::split_hash(&identity(current), split, &rows),
        );
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
    let train: BTreeSet<_> = splits["train"].iter().map(|e| e.prompt.clone()).collect();
    let names: Vec<_> = splits
        .keys()
        .filter(|s| s.starts_with("validation"))
        .cloned()
        .collect();
    let mut empty = vec![];
    for name in names {
        let novel: Vec<_> = splits[&name]
            .iter()
            .filter(|e| !train.contains(&e.prompt))
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

pub(crate) fn uses_numbers(spec: &Value) -> bool {
    spec["task"] == "histogram"
        || spec["task"] == "histogram2"
        || (spec["task"] == "mode" && spec["scratchpad"] != "none")
}
/// Source Generator.identity omits the unused number limit for Mode(none).
pub(crate) fn identity(spec: &Value) -> Value {
    let mut identity = spec.clone();
    if !uses_numbers(spec) {
        let object = identity.as_object_mut().expect("validated specification");
        object.remove("number_limit");
    }
    identity
}

pub fn metadata(data: &Dataset) -> Value {
    let mut out = data.metadata();
    out["generation"] = json!(data.task.splits.iter().map(|(name, rows)| {
        (name, json!({"examples":rows.len(),"answer_tokens_including_eos":rows.iter().map(|e|e.answer.len()).sum::<usize>(),
            "max_prompt":rows.iter().map(|e|e.prompt.len()).max(),"max_answer":rows.iter().map(|e|e.answer.len()).max(),
            "max_generation_limit":rows.iter().map(|e|e.limit).max(),
            "complete_records_sha256":hash(data::json_text(&json!(rows.iter().map(wire::value).collect::<Vec<_>>())).as_bytes())}))
    }).collect::<BTreeMap<_,_>>());
    out["scoring"] = json!(
        "greedy free generation from prompt, own predictions through EOS/limit; teacher-forced CE separate"
    );
    out["final_token_diagnostic"] = json!(data.task.splits["train"][0].task == "mode");
    let observed: BTreeSet<_> = data.task.splits["train"]
        .iter()
        .flat_map(|e| e.answer.iter().copied())
        .collect();
    out["output_label_coverage"] = json!(
        data.task
            .splits
            .iter()
            .map(|(name, rows)| {
                let mut missing = BTreeMap::<i64, usize>::new();
                for token in rows.iter().flat_map(|e| &e.answer) {
                    if !observed.contains(token) {
                        *missing.entry(*token).or_default() += 1;
                    }
                }
                (
                    name,
                    json!({"unseen_target_occurrences":missing.values().sum::<usize>(),
            "unseen_raw_labels":missing,"training_output_classes":observed.len(),
            "identifies_missing_labels":false}),
                )
            })
            .collect::<BTreeMap<_, _>>()
    );
    out
}

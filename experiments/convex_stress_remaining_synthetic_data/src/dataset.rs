//! Source seed1 pools, task-specific probes, masks and novelty rules.
use crate::*;
pub use generator::generate;
pub use sequences::dataset::Dataset;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

pub fn specification(case: &str) -> Result<(Value, Vec<usize>)> {
    if !CASES.contains(&case) {
        return Err("case outside remaining synthetic cohort".into());
    }
    let p: Value = serde_json::from_str(PROTOCOL)?;
    let mut s = p["definitions"][case].clone();
    let lengths = s
        .as_object_mut()
        .ok_or("missing definition")?
        .remove("longer_lengths")
        .ok_or("missing probes")?;
    Ok((
        s,
        lengths
            .as_array()
            .ok_or("invalid probes")?
            .iter()
            .map(|x| usize::try_from(x.as_u64().ok_or("invalid probe length")?).map_err(Into::into))
            .collect::<Result<_>>()?,
    ))
}
/// Source Memorization uses Generator.key. RandomLM overrides it with the
/// entire sampled payload; ordinary generation uses its raw prompt.
pub fn key(e: &data::Example) -> &[i64] {
    if e.task == "random_lm" || e.prompt.is_empty() {
        &e.tokens
    } else {
        &e.prompt
    }
}
pub fn build(case: &str) -> Result<Dataset> {
    let (s, lengths) = specification(case)?;
    let random = s["task"] == "random_lm";
    let (train_count, validation_count, test_count) =
        if random { (8, 64, 128) } else { (128, 32, 64) };
    let mut splits: BTreeMap<String, Vec<data::Example>> = BTreeMap::new();
    let mut fingerprints: BTreeMap<String, String> = BTreeMap::new();
    let mut add = |name: &str, split: &str, count: usize, current: &Value| -> Result<()> {
        let rows = generate(current, split, count)?;
        fingerprints.insert(name.into(), wire::split_hash(current, split, &rows));
        splits.insert(name.into(), rows);
        Ok(())
    };
    for (split, count) in [
        ("train", train_count),
        ("validation", validation_count),
        ("test", test_count),
    ] {
        add(split, split, count, &s)?;
    }
    add("train_clean", "train", train_count, &s)?;
    let mut probes = vec![];
    for &length in &lengths {
        let mut probe = s.clone();
        probe["length"] = json!(length);
        probe["min_length"] = Value::Null;
        probes.push((format!("length-{length}"), probe));
    }
    if s["task"] == "addition" {
        for length in std::iter::once(spec::number(&s, "length")?).chain(lengths.iter().copied()) {
            let mut probe = s.clone();
            probe["length"] = json!(length);
            probe["min_length"] = Value::Null;
            probe["carry_length"] = json!(length);
            probes.push((format!("hard-carry-length-{length}"), probe));
        }
    }
    for (name, probe) in probes {
        for (split, count) in [("validation", validation_count), ("test", test_count)] {
            add(&format!("{split}/{name}"), split, count, &probe)?;
        }
    }
    let train: BTreeSet<_> = splits["train"].iter().map(|e| key(e).to_vec()).collect();
    let names: Vec<_> = splits
        .keys()
        .filter(|s| s.starts_with("validation"))
        .cloned()
        .collect();
    let mut empty = vec![];
    for name in names {
        let novel: Vec<_> = splits[&name]
            .iter()
            .filter(|e| !train.contains(key(e)))
            .cloned()
            .collect();
        let dest = format!("{name}/novel");
        if novel.is_empty() {
            empty.push(dest);
        } else {
            splits.insert(dest, novel);
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
            vocab: spec::vocabulary(&s)?,
            selection: selection.into(),
            splits,
        },
        original_fingerprints: fingerprints,
        tensor_fingerprints,
        empty_novel_splits: empty,
    })
}
pub fn metadata(d: &Dataset) -> Result<Value> {
    let mut out = d.metadata();
    let (s, _) = specification(&d.name)?;
    let prefix = s["task"] == "crasp";
    out["scoring"] = json!(if prefix {
        "all supervised positions of each input prefix; no generated answer"
    } else {
        "own-output generation through EOS or the input-derived limit; teacher-forced CE separate"
    });
    out["novelty_key"] = json!(if s["task"] == "random_lm" {
        "full sampled teacher-forced sequence"
    } else if prefix {
        "complete input sequence"
    } else {
        "raw prompt"
    });
    out["deterministic_target"] = json!(s["task"] != "random_lm");
    out["final_answer_is_exact"] = json!(true);
    if prefix {
        let program = crasp::select(
            spec::number(&s, "formula_depth")?,
            spec::number(&s, "formula_seed")? as u32,
        )?;
        out["data_oracle_only"] = json!({"selected":program,"description":program.formula.describe(),
            "syntactic_depth":program.formula.depth(),"available_to_predictor":false});
    }
    out["records"]=json!(d.task.splits.iter().map(|(name,rows)|(name,json!({
        "examples":rows.len(),"target_count":rows.iter().map(|e|e.targets.iter().filter(|&&y|y>=0).count()).sum::<usize>(),
        "transition_count":rows.iter().map(|e|e.changes.iter().filter(|&&b|b).count()).sum::<usize>(),
        "max_prompt":rows.iter().map(|e|e.prompt.len()).max(),"max_answer":rows.iter().map(|e|e.answer.len()).max(),
        "max_generation_limit":rows.iter().map(|e|e.limit).max(),
        "complete_records_sha256":hash(data::json_text(&json!(rows.iter().map(wire::value).collect::<Vec<_>>())).as_bytes())
    }))).collect::<BTreeMap<_,_>>());
    let observed: BTreeSet<_> = d.task.splits["train"]
        .iter()
        .flat_map(|e| e.targets.iter())
        .filter(|&&y| y >= 0)
        .copied()
        .collect();
    out["output_label_coverage"]=json!(d.task.splits.iter().map(|(name,rows)|{
        let mut missing=BTreeMap::<i64,usize>::new();
        for &y in rows.iter().flat_map(|e|e.targets.iter()).filter(|&&y|y>=0) {
            if !observed.contains(&y) {*missing.entry(y).or_default()+=1;}
        }
        (name,json!({"unseen_target_occurrences":missing.values().sum::<usize>(),
            "unseen_raw_labels":missing,"training_output_classes":observed.len(),"identifies_missing_labels":false}))
    }).collect::<BTreeMap<_,_>>());
    Ok(out)
}

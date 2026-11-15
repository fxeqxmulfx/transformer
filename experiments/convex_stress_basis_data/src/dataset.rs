//! Preserve Basis's full source pools, original128-length selection, novel
//! subsets, and the noisy study's disjoint clean pools. No subsampling.
use crate::*;
pub use sequences::dataset::Dataset;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

pub fn specification(case: &str) -> Result<Value> {
    if !CASES.contains(&case) {
        return Err("case outside remaining Basis cohort".into());
    }
    let p: Value = serde_json::from_str(PROTOCOL)?;
    Ok(p["definitions"][case].clone())
}
pub fn build(case: &str) -> Result<(Dataset, Value)> {
    let s = specification(case)?;
    let noisy = s["task"] == "parity";
    let language = s["task"] == "blocks";
    let sizes = if noisy {
        [512, 128, 256]
    } else {
        [20000, 512, 512]
    };
    let mut splits = BTreeMap::new();
    let mut fingerprints = BTreeMap::new();
    let mut seen = BTreeSet::new();
    for (name, count) in ["train", "validation", "test"].into_iter().zip(sizes) {
        let rows = generator::generate(&s, name, count, &seen, noisy)?;
        if noisy {
            seen.extend(rows.iter().map(|e| generator::key(e).to_vec()));
        }
        fingerprints.insert(name.into(), wire::split_hash(&s, name, &rows));
        splits.insert(name.into(), rows);
    }
    let mut noise_report = Value::Null;
    if noisy {
        let clean = splits["train"].clone();
        let (observed, report) = noise::apply(&clean, 0.2, 2)?;
        fingerprints.insert("train_clean".into(), wire::split_hash(&s, "train", &clean));
        fingerprints.insert("train".into(), wire::split_hash(&s, "train", &observed));
        splits.insert("train_clean".into(), clean);
        splits.insert("train".into(), observed);
        noise_report = report;
    }
    if language {
        let mut probe = s.clone();
        probe["length"] = json!(128);
        probe["min_length"] = Value::Null;
        for name in ["validation", "test"] {
            let rows = generator::generate(&probe, name, 512, &BTreeSet::new(), false)?;
            fingerprints.insert(
                format!("{name}/length-128"),
                wire::split_hash(&probe, name, &rows),
            );
            splits.insert(format!("{name}/length-128"), rows);
        }
    }
    let original = if language {
        "validation/length-128"
    } else {
        "validation"
    };
    let train: BTreeSet<_> = splits["train"]
        .iter()
        .map(|e| generator::key(e).to_vec())
        .collect();
    let novel: Vec<_> = splits[original]
        .iter()
        .filter(|e| !train.contains(generator::key(e)))
        .cloned()
        .collect();
    let dest = format!("{original}/novel");
    let mut empty = vec![];
    let selection = if novel.is_empty() {
        empty.push(dest);
        original.to_string()
    } else {
        splits.insert(dest.clone(), novel);
        dest
    };
    let tensor_fingerprints = splits
        .iter()
        .map(|(name, rows)| (name.clone(), wire::tensor_hash(rows)))
        .collect();
    // Basis uses Parity's default32 symbols (68 raw classes), unlike the
    // suite parity cases that explicitly supply64 symbols and100 classes.
    let vocab = if s["task"] == "mqar" {
        36 + 2 * generator::number(&s, "symbols")?
    } else if noisy {
        68
    } else {
        36
    };
    Ok((
        Dataset {
            name: case.into(),
            task: data::Task {
                splits,
                vocab,
                selection,
            },
            original_fingerprints: fingerprints,
            tensor_fingerprints,
            empty_novel_splits: empty,
        },
        noise_report,
    ))
}
pub fn metadata(d: &Dataset, noise: &Value) -> Result<Value> {
    let s = specification(&d.name)?;
    let mut out = d.metadata();
    out["noise"] = noise.clone();
    out["data_seed"] = json!(1);
    out["new_model_evaluations"] = json!(0);
    out["original_selection"] = json!(if s["task"] == "blocks" {
        "validation/length-128"
    } else {
        "validation"
    });
    out["clean_labels_available_to_predictor"] = json!(false);
    out["records"]=json!(d.task.splits.iter().map(|(name,rows)|(name,json!({
        "examples":rows.len(),"target_count":rows.iter().map(|e|e.targets.iter().filter(|&&y|y>=0).count()).sum::<usize>(),
        "transition_count":rows.iter().map(|e|e.changes.iter().filter(|&&b|b).count()).sum::<usize>(),
        "complete_records_sha256":hash(data::json_text(&json!(rows.iter().map(wire::value).collect::<Vec<_>>())).as_bytes())
    }))).collect::<BTreeMap<_,_>>());
    Ok(out)
}

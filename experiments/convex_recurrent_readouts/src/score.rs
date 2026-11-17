//! Original scoring only after all four readout-refit endpoints are sealed.
use crate::{
    Result, continuation,
    experiment::{self, cell, settings_with},
    source,
};
use bilinear::{experiment as original, guard};
use rotary::model::Model;
use serde_json::{Value, json};
use std::{fs, path::Path};

pub fn sealed(cohort: &Path) -> Result<()> {
    sealed_with(cohort, experiment::PROTOCOL)
}
pub fn sealed_with(cohort: &Path, text: &str) -> Result<()> {
    let binary = original::hash(&fs::read(std::env::current_exe()?)?);
    for variant in experiment::VARIANTS {
        for mode in experiment::MODES {
            let (protocol, _) = settings_with(variant, mode, text)?;
            let path = cell(cohort, variant, mode);
            let meta = original::score::read(&path.join("metadata.json"))?;
            let receipt = original::score::read(&path.join("training_receipt.json"))?;
            if meta["source_fingerprint"] != experiment::fingerprint()
                || meta["protocol"] != protocol
                || meta["binary_sha256"] != binary
                || meta["variant"] != variant
                || meta["mode"] != mode
                || receipt["source_fingerprint"] != experiment::fingerprint()
                || receipt["variant"] != variant
                || receipt["mode"] != mode
                || receipt["status"] != "sealed"
                || receipt["heldout_scored"] != false
                || receipt["updates"] != protocol["updates"]
                || receipt["content_sha256"] != meta["input"]["content_sha256"]
                || receipt["endpoint_sha256"]
                    != original::hash(&fs::read(path.join("endpoint.json"))?)
            {
                return Err("incomplete or changed refit cohort seal".into());
            }
            continuation::verify_seal(&path, &protocol, &meta)?;
        }
    }
    Ok(())
}

pub fn run(
    workspace: &Path,
    storage: &Path,
    cohort: &Path,
    variant: &str,
    mode: &str,
    replay: bool,
    text: &str,
) -> Result<()> {
    guard::admit(0, workspace, storage)?;
    sealed_with(cohort, text)?;
    settings_with(variant, mode, text)?;
    let path = cell(cohort, variant, mode);
    let meta = original::score::read(&path.join("metadata.json"))?;
    let (_, row, task) = original::case_in(source::CASE, rotary::experiment::CAUTIOUS_PROTOCOL)?;
    if meta["original"] != row
        || meta["objects"] != task.vocab
        || meta["selection"] != task.selection
    {
        return Err("original scoring identity changed".into());
    }
    let endpoint = path.join("endpoint.json");
    guard::admit(
        fs::metadata(&endpoint)?
            .len()
            .checked_mul(5)
            .ok_or("score reservation")?,
        workspace,
        storage,
    )?;
    let bytes = fs::read(endpoint)?;
    let model: Model = serde_json::from_slice(&bytes)?;
    model.validate()?;
    if source::content_hash(&model)?
        != meta["input"]["content_sha256"]
            .as_str()
            .ok_or("content hash")?
        || model.budget != 16.
        || model.objects != task.vocab
        || model.atoms.len() != 129
    {
        return Err("refit endpoint capacity/content mismatch".into());
    }
    let mut splits = serde_json::Map::new();
    for (name, examples) in &task.splits {
        guard::admit(0, workspace, storage)?;
        let report = rotary::experiment::score::evaluate(&model, examples)?;
        println!(
            "source={variant} mode={mode} split={name} raw={} exact={} ce={}",
            report["metrics"]["token_accuracy"],
            report["metrics"]["sequence_accuracy"],
            report["metrics"]["loss"]
        );
        splits.insert(name.clone(), report);
    }
    let metric = row["case"]["metric"].as_str().ok_or("metric")?;
    let threshold = row["case"]["target"].as_f64().ok_or("threshold")?;
    let selected = &splits[&task.selection]["metrics"];
    let result = json!({"source_fingerprint":experiment::fingerprint(),"endpoint_sha256":original::hash(&bytes),
        "variant":variant,"mode":mode,"selection":task.selection,"original_metric":metric,"original_threshold":threshold,
        "original_success":selected[metric].as_f64().ok_or("metric result")?>=threshold,
        "raw_and_exact_99":selected["token_accuracy"].as_f64().ok_or("raw")?>=0.99&&selected["sequence_accuracy"].as_f64().ok_or("exact")?>=0.99,
        "splits":splits});
    if replay {
        let previous: Value = original::score::read(&path.join("scores.json"))?;
        if result != previous {
            return Err("complete refit raw-logit replay differs".into());
        }
        original::write_json(
            &path.join("replay.json"),
            &json!({"status":"identical",
            "source_fingerprint":experiment::fingerprint(),"endpoint_sha256":original::hash(&bytes),
            "score_sha256":original::hash(&fs::read(path.join("scores.json"))?)}),
        )?;
    } else {
        original::write_json(&path.join("scores.json"), &result)?;
    }
    Ok(())
}

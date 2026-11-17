//! Original complete scoring after the entire fixed cohort seals; exact replay.
use crate::{Result, experiment};
use bilinear::{experiment as original, guard};
use rotary::model::Model;
use serde_json::{Value, json};
use std::{fs, path::Path};

pub fn sealed(cohort: &Path) -> Result<()> {
    let binary = original::hash(&fs::read(std::env::current_exe()?)?);
    let pins: Vec<Value> = serde_json::from_str(optimality::source::INPUTS)?;
    for variant in experiment::VARIANTS {
        for mode in experiment::MODES {
            let (p, _) = experiment::settings(variant, mode)?;
            let path = experiment::cell(cohort, variant, mode);
            let meta = original::score::read(&path.join("metadata.json"))?;
            let r = original::score::read(&path.join("training_receipt.json"))?;
            let expected = pins
                .iter()
                .find(|v| {
                    v["cohort"] == p["training_input_cohort"]
                        && v["variant"] == variant
                        && v["mode"] == mode
                })
                .ok_or("input pin")?;
            if meta["source_fingerprint"] != experiment::fingerprint()
                || meta["protocol"] != p
                || meta["binary_sha256"] != binary
                || meta["input"] != *expected
                || meta["variant"] != variant
                || meta["mode"] != mode
                || r["source_fingerprint"] != experiment::fingerprint()
                || r["variant"] != variant
                || r["mode"] != mode
                || r["updates"] != p["updates"]
                || r["status"] != "sealed"
                || r["heldout_scored"] != false
                || r["content_sha256"] != expected["content_sha256"]
                || r["endpoint_sha256"] != original::hash(&fs::read(path.join("endpoint.json"))?)
            {
                return Err("incomplete or changed joint-capacity cohort seal".into());
            }
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
) -> Result<()> {
    guard::admit(0, workspace, storage)?;
    sealed(cohort)?;
    experiment::settings(variant, mode)?;
    let path = experiment::cell(cohort, variant, mode);
    let meta = original::score::read(&path.join("metadata.json"))?;
    let (_, row, task) = original::case_in(
        readouts::source::CASE,
        rotary::experiment::CAUTIOUS_PROTOCOL,
    )?;
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
            .checked_mul(6)
            .ok_or("score reservation")?,
        workspace,
        storage,
    )?;
    let bytes = fs::read(endpoint)?;
    let model: Model = serde_json::from_slice(&bytes)?;
    model.validate()?;
    if readouts::source::content_hash(&model)?
        != meta["input"]["content_sha256"]
            .as_str()
            .ok_or("content hash")?
        || model.objects != task.vocab
        || model.budget != 16.
        || model.atoms.len() != 129
    {
        return Err("joint-capacity endpoint content changed".into());
    }
    let mut splits = serde_json::Map::new();
    for (name, examples) in &task.splits {
        guard::admit(0, workspace, storage)?;
        let report = rotary::experiment::score::evaluate(&model, examples)?;
        println!(
            "{variant}/{mode} split={name} raw={} exact={} ce={}",
            report["metrics"]["token_accuracy"],
            report["metrics"]["sequence_accuracy"],
            report["metrics"]["loss"]
        );
        splits.insert(name.clone(), report);
    }
    let metric = row["case"]["metric"].as_str().ok_or("metric")?;
    let threshold = row["case"]["target"].as_f64().ok_or("threshold")?;
    let selected = &splits[&task.selection]["metrics"];
    let report = json!({"source_fingerprint":experiment::fingerprint(),"endpoint_sha256":original::hash(&bytes),
        "variant":variant,"mode":mode,"selection":task.selection,"original_metric":metric,"original_threshold":threshold,
        "original_success":selected[metric].as_f64().ok_or("primary metric")?>=threshold,
        "raw_and_exact_99":selected["token_accuracy"].as_f64().ok_or("raw")?>=0.99 &&
            selected["sequence_accuracy"].as_f64().ok_or("exact")?>=0.99,"splits":splits});
    if replay {
        if report != original::score::read(&path.join("scores.json"))? {
            return Err("complete raw replay differs".into());
        }
        original::write_json(
            &path.join("replay.json"),
            &json!({"status":"identical",
            "source_fingerprint":experiment::fingerprint(),"endpoint_sha256":original::hash(&bytes),
            "score_sha256":original::hash(&fs::read(path.join("scores.json"))?)}),
        )?;
    } else {
        original::write_json(&path.join("scores.json"), &report)?;
    }
    Ok(())
}

//! Original full-class scoring after the whole cohort seals, plus exact replay.
use crate::{Result, experiment, source};
use bilinear::{experiment as original, guard};
use rotary::model::Model;
use serde_json::{Value, json};
use std::{fs, path::Path};

pub fn sealed(cohort: &Path) -> Result<()> {
    let binary = original::hash(&fs::read(std::env::current_exe()?)?);
    for arm in experiment::ARMS {
        for variant in experiment::VARIANTS {
            for mode in experiment::MODES {
                let (p, _, _, _) = experiment::settings(arm, variant, mode)?;
                let path = experiment::cell(cohort, arm, variant, mode);
                let meta = original::score::read(&path.join("metadata.json"))?;
                let r = original::score::read(&path.join("training_receipt.json"))?;
                let expected = source::pin(variant, mode)?;
                if meta["source_fingerprint"] != experiment::fingerprint()
                    || meta["protocol"] != p
                    || meta["binary_sha256"] != binary
                    || meta["input"] != expected
                    || meta["arm"] != arm
                    || meta["variant"] != variant
                    || meta["mode"] != mode
                    || r["source_fingerprint"] != experiment::fingerprint()
                    || r["arm"] != arm
                    || r["variant"] != variant
                    || r["mode"] != mode
                    || r["updates"] != p["updates"]
                    || r["status"] != "sealed"
                    || r["heldout_scored"] != false
                    || r["initial_content_sha256"] != meta["initial_content_sha256"]
                    || r["endpoint_sha256"]
                        != original::hash(&fs::read(path.join("endpoint.json"))?)
                {
                    return Err("incomplete or changed content cohort seal".into());
                }
                if r["cache_sha256"] != original::hash(&fs::read(path.join("cache.json"))?)
                    || r["history_sha256"] != original::hash(&fs::read(path.join("history.jsonl"))?)
                {
                    return Err("historical cache or content history seal changed".into());
                }
            }
        }
    }
    Ok(())
}
pub fn load(
    workspace: &Path,
    storage: &Path,
    path: &Path,
) -> Result<(Model, Value, baseline::data::Task)> {
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
    let model: Model = serde_json::from_slice(&fs::read(endpoint)?)?;
    model.validate()?;
    let receipt = original::score::read(&path.join("training_receipt.json"))?;
    if readouts::source::content_hash(&model)?
        != receipt["content_sha256"].as_str().ok_or("content hash")?
        || model.objects != task.vocab
        || model.budget != 16.
        || model.atoms.len() != 129
        || model.family != rotary::atom::Family::Rotary
        || (meta["arm"] == "control" && receipt["content_sha256"] != meta["initial_content_sha256"])
    {
        return Err("complete endpoint family or content changed".into());
    }
    Ok((model, row, task))
}
pub fn run(
    workspace: &Path,
    storage: &Path,
    cohort: &Path,
    arm: &str,
    variant: &str,
    mode: &str,
    replay: bool,
) -> Result<()> {
    guard::admit(0, workspace, storage)?;
    sealed(cohort)?;
    experiment::settings(arm, variant, mode)?;
    let path = experiment::cell(cohort, arm, variant, mode);
    let (model, row, task) = load(workspace, storage, &path)?;
    let endpoint_hash = original::hash(&fs::read(path.join("endpoint.json"))?);
    let mut splits = serde_json::Map::new();
    for (name, examples) in &task.splits {
        guard::admit(0, workspace, storage)?;
        let mut report = rotary::experiment::score::evaluate(&model, examples)?;
        report["raw_and_exact_99"] = json!(
            report["metrics"]["token_accuracy"].as_f64().ok_or("raw")? >= 0.99
                && report["metrics"]["sequence_accuracy"]
                    .as_f64()
                    .ok_or("exact")?
                    >= 0.99
        );
        println!(
            "{arm}/{variant}/{mode} split={name} raw={} exact={} ce={}",
            report["metrics"]["token_accuracy"],
            report["metrics"]["sequence_accuracy"],
            report["metrics"]["loss"]
        );
        splits.insert(name.clone(), report);
    }
    let metric = row["case"]["metric"].as_str().ok_or("metric")?;
    let threshold = row["case"]["target"].as_f64().ok_or("threshold")?;
    let selected = &splits[&task.selection];
    let report = json!({"source_fingerprint":experiment::fingerprint(),"endpoint_sha256":endpoint_hash,
        "arm":arm,"variant":variant,"mode":mode,"selection":task.selection,
        "original_metric":metric,"original_threshold":threshold,
        "original_success":selected["metrics"][metric].as_f64().ok_or("primary metric")? >= threshold,
        "raw_and_exact_99":selected["raw_and_exact_99"],"splits":splits});
    if replay {
        if report != original::score::read(&path.join("scores.json"))? {
            return Err("complete raw replay differs".into());
        }
        original::write_json(
            &path.join("replay.json"),
            &json!({"status":"identical",
            "source_fingerprint":experiment::fingerprint(),"endpoint_sha256":endpoint_hash,
            "score_sha256":original::hash(&fs::read(path.join("scores.json"))?)}),
        )?;
    } else {
        original::write_json(&path.join("scores.json"), &report)?;
    }
    Ok(())
}

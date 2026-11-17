//! Training-only state inspection of immutable, previously scored checkpoints.
//! CM98 bounds actual rotary execution; these measurements add no learning claim.
use crate::{Result, atom::Family, experiment, model::Model};
use bilinear::{experiment as original, guard};
use serde_json::json;
use std::{fs, path::Path};
mod measure;
#[cfg(test)]
mod tests;
mod trace;

pub fn fingerprint() -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    for source in [
        include_bytes!("retention.rs").as_slice(),
        include_bytes!("retention/measure.rs"),
        include_bytes!("retention/trace.rs"),
        include_bytes!("bin/retention.rs"),
        experiment::fingerprint().as_bytes(),
    ] {
        h.update((source.len() as u64).to_le_bytes());
        h.update(source);
    }
    h.finalize().iter().map(|x| format!("{x:02x}")).collect()
}

pub fn run(
    workspace: &Path,
    storage: &Path,
    cohort: &Path,
    output: &Path,
    variant: &str,
    mode: &str,
) -> Result<()> {
    guard::admit(0, workspace, storage)?;
    let (text, producer) = match variant {
        "cautious" => (experiment::CAUTIOUS_PROTOCOL, crate::diagnostic::PRODUCER),
        "refined" => (
            experiment::REFINED_PROTOCOL,
            "0f0552315bfd778e2f10ac1aac56cd1593ed4b3aa284a85ae46c2e787ebb859f",
        ),
        _ => return Err("unknown sealed retention cohort".into()),
    };
    let (protocol, _) = experiment::protocol(mode, 1, text)?;
    let name = "gpt-probe-key-value-binding";
    let (case, original_row, task) = original::case_in(name, text)?;
    let cell = experiment::score::cell(cohort, name, mode, 1, Family::Rotary);
    let meta = original::score::read(&cell.join("metadata.json"))?;
    let receipt = original::score::read(&cell.join("training_receipt.json"))?;
    let endpoint = cell.join("endpoint.json");
    let reservation = fs::metadata(&endpoint)?
        .len()
        .checked_mul(5)
        .and_then(|n| n.checked_add(128 * 1024 * 1024))
        .ok_or("retention allocation overflow")?;
    guard::admit(reservation, workspace, storage)?;
    let bytes = fs::read(&endpoint)?;
    if meta["source_fingerprint"] != producer
        || receipt["source_fingerprint"] != producer
        || meta["protocol"] != protocol
        || meta["original"] != original_row
        || meta["case"] != case
        || meta["mode"] != mode
        || meta["seed"] != 1
        || meta["family"] != "rotary"
        || receipt["status"] != "sealed"
        || receipt["updates"] != 128
        || receipt["endpoint_sha256"] != original::hash(&bytes)
    {
        return Err("retention input is not the declared immutable producer".into());
    }
    let model: Model = serde_json::from_slice(&bytes)?;
    model.validate()?;
    if model.objects != task.vocab
        || model.family != Family::Rotary
        || model.atoms.len() != 129
        || model.budget != protocol["budget"].as_f64().ok_or("budget")?
    {
        return Err("retention checkpoint capacity mismatch".into());
    }
    let training = &task.splits["train"];
    let records = original::records(training, &(0..training.len()).collect::<Vec<_>>())?;
    let mut inventory = vec![];
    for r in &records {
        inventory.extend((r.prefix.len() as u64).to_le_bytes());
        for &o in r.prefix {
            inventory.extend(o.to_le_bytes());
        }
        inventory.extend((r.target as u64).to_le_bytes());
    }
    let report = measure::run(&model, &records, || {
        guard::admit(reservation, workspace, storage).map(|_| ())
    })?;
    fs::create_dir(output)?;
    original::write_json(
        &output.join("metadata.json"),
        &json!({
            "diagnostic_fingerprint":fingerprint(),"producer_fingerprint":producer,
            "endpoint_sha256":original::hash(&bytes),"variant":variant,"mode":mode,
            "original_training_examples":training.len(),"records":records.len(),
            "original":original_row,"training_records_sha256":original::hash(&inventory),
            "classes":model.objects,"heldout_labels_used":false,"model_update_accepted":false,
            "feature_collision_comparison":"every weight-scaled complex state coordinate; exact f64 values, signed zeros equal",
            "scope":"original training split; every stored component and every raw input occurrence"
        }),
    )?;
    original::write_json(&output.join("states.json"), &report)?;
    println!("{}", report["summary"]);
    Ok(())
}

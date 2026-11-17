//! Immutable checkpoint/data adapter. Source names never enter model inference.
use crate::Result;
use baseline::data::Task;
use bilinear::{experiment as original, guard};
use rotary::{atom::Family, model::Model};
use serde_json::{Value, json};
use std::{fs, path::Path};
pub const CASE: &str = "gpt-probe-key-value-binding";

pub fn content_hash(model: &Model) -> Result<String> {
    let content: Vec<_> = model
        .atoms
        .iter()
        .map(|a| (&a.frequency, &a.initial, &a.transition))
        .collect();
    Ok(original::hash(&serde_json::to_vec(&content)?))
}
pub fn load(
    workspace: &Path,
    storage: &Path,
    sources: &Path,
    variant: &str,
    mode: &str,
) -> Result<(Model, Value, Value, Task)> {
    let (text, producer) = match variant {
        "cautious" => (
            rotary::experiment::CAUTIOUS_PROTOCOL,
            rotary::diagnostic::PRODUCER,
        ),
        "refined" => (
            rotary::experiment::REFINED_PROTOCOL,
            "0f0552315bfd778e2f10ac1aac56cd1593ed4b3aa284a85ae46c2e787ebb859f",
        ),
        _ => return Err("unknown immutable source".into()),
    };
    let (protocol, _) = rotary::experiment::protocol(mode, 1, text)?;
    let (case, row, task) = original::case_in(CASE, text)?;
    let cell = rotary::experiment::score::cell(
        &sources.join(format!("{variant}-v1")),
        CASE,
        mode,
        1,
        Family::Rotary,
    );
    let meta = original::score::read(&cell.join("metadata.json"))?;
    let receipt = original::score::read(&cell.join("training_receipt.json"))?;
    let endpoint = cell.join("endpoint.json");
    guard::admit(
        fs::metadata(&endpoint)?
            .len()
            .checked_mul(6)
            .ok_or("source reservation")?,
        workspace,
        storage,
    )?;
    let raw = fs::read(endpoint)?;
    let hash = original::hash(&raw);
    if meta["protocol"] != protocol
        || meta["source_fingerprint"] != producer
        || meta["original"] != row
        || meta["case"] != case
        || meta["family"] != "rotary"
        || meta["mode"] != mode
        || meta["seed"] != 1
        || receipt["status"] != "sealed"
        || receipt["source_fingerprint"] != producer
        || receipt["endpoint_sha256"] != hash
        || receipt["updates"] != 128
    {
        return Err("source is not the declared sealed learned model".into());
    }
    let model: Model = serde_json::from_slice(&raw)?;
    model.validate()?;
    if model.objects != task.vocab
        || model.budget != 16.
        || model.family != Family::Rotary
        || model.atoms.len() != 129
    {
        return Err("source model capacity changed".into());
    }
    let source = json!({"variant":variant,"mode":mode,"source_fingerprint":producer,
        "endpoint_sha256":hash,"content_sha256":content_hash(&model)?,"components":model.atoms.len()});
    Ok((model, row, source, task))
}

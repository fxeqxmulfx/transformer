//! Pinned immutable original-training inputs. No score files are opened.
use crate::Result;
use baseline::data::Task;
use bilinear::{experiment as original, guard};
use rotary::model::Model;
use serde_json::Value;
use std::{fs, path::Path};
pub const INPUTS: &str = include_str!("../inputs.json");

pub fn load(
    workspace: &Path,
    storage: &Path,
    sources: &Path,
    cohort: &str,
    variant: &str,
    mode: &str,
) -> Result<(Model, Value, Task)> {
    let inputs: Vec<Value> = serde_json::from_str(INPUTS)?;
    let expected = inputs
        .iter()
        .find(|v| v["cohort"] == cohort && v["variant"] == variant && v["mode"] == mode)
        .ok_or("unlisted immutable diagnostic input")?;
    let text = match cohort {
        "original-v1" => readouts::experiment::PROTOCOL,
        "long-v1" => readouts::experiment::LONG_PROTOCOL,
        "projected-v1" => readouts::experiment::PROJECTED_PROTOCOL,
        _ => return Err("unknown producer cohort".into()),
    };
    let (protocol, _) = readouts::experiment::settings_with(variant, mode, text)?;
    let (_, row, input, task) = readouts::source::load(workspace, storage, sources, variant, mode)?;
    let path = storage
        .join("transformer-runs/recurrent-readouts")
        .join(cohort)
        .join(variant)
        .join(mode);
    let meta = original::score::read(&path.join("metadata.json"))?;
    let receipt = original::score::read(&path.join("training_receipt.json"))?;
    let endpoint = path.join("endpoint.json");
    guard::admit(
        fs::metadata(&endpoint)?
            .len()
            .checked_mul(6)
            .ok_or("input reservation")?,
        workspace,
        storage,
    )?;
    let raw = fs::read(endpoint)?;
    let hash = original::hash(&raw);
    if meta["protocol"] != protocol
        || meta["original"] != row
        || meta["input"] != input
        || meta["source_fingerprint"] != expected["source_fingerprint"]
        || meta["binary_sha256"] != expected["binary_sha256"]
        || receipt["source_fingerprint"] != expected["source_fingerprint"]
        || receipt["endpoint_sha256"] != hash
        || expected["endpoint_sha256"] != hash
        || receipt["content_sha256"] != expected["content_sha256"]
        || receipt["updates"] != expected["updates"]
        || receipt["status"] != "sealed"
        || meta["variant"] != variant
        || meta["mode"] != mode
    {
        return Err("immutable diagnostic producer or checkpoint mismatch".into());
    }
    let model: Model = serde_json::from_slice(&raw)?;
    model.validate()?;
    if model.objects != task.vocab
        || model.objects != 100
        || model.budget != 16.
        || model.atoms.len() != 129
        || readouts::source::content_hash(&model)? != input["content_sha256"]
    {
        return Err("complete diagnostic family/content mismatch".into());
    }
    Ok((model, expected.clone(), task))
}

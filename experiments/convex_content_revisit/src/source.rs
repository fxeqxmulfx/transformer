//! Immutable training inputs; no score file is opened by training.
use crate::Result;
use bilinear::{experiment as original, guard};
use rotary::model::Model;
use serde_json::Value;
use std::{fs, path::Path};
pub const INPUTS: &str = include_str!("../inputs.json");
pub const PRODUCER: &str = "738b1d49942d9daebb6665bde271683c073f13d3c0ede37485b5b879b4676905";
pub const BINARY: &str = "4b7e4056acf755feccefb7d25e01e147e2fa688fa28fccf6287a93464d8caabe";
pub fn pin(variant: &str, mode: &str) -> Result<Value> {
    let pins: Vec<Value> = serde_json::from_str(INPUTS)?;
    pins.into_iter()
        .find(|p| p["variant"] == variant && p["mode"] == mode)
        .ok_or_else(|| "unlisted content input".into())
}
pub fn load(
    workspace: &Path,
    storage: &Path,
    variant: &str,
    mode: &str,
) -> Result<(Model, retained::cache::Cache, Value, baseline::data::Task)> {
    guard::admit(0, workspace, storage)?;
    let expected = pin(variant, mode)?;
    let path = storage
        .join("transformer-runs/retained-readouts/original-v1/retained-small")
        .join(variant)
        .join(mode);
    let reservation = fs::metadata(path.join("endpoint.json"))?
        .len()
        .checked_add(fs::metadata(path.join("cache.json"))?.len())
        .ok_or("input size")?
        .checked_mul(8)
        .ok_or("input reservation")?;
    guard::admit(reservation, workspace, storage)?;
    for (name, key) in [
        ("metadata.json", "metadata_sha256"),
        ("training_receipt.json", "receipt_sha256"),
        ("endpoint.json", "endpoint_sha256"),
        ("cache.json", "cache_sha256"),
    ] {
        if expected[key] != original::hash(&fs::read(path.join(name))?) {
            return Err("immutable content input changed".into());
        }
    }
    let meta = original::score::read(&path.join("metadata.json"))?;
    let receipt = original::score::read(&path.join("training_receipt.json"))?;
    let (_, row, task) = original::case_in(
        readouts::source::CASE,
        rotary::experiment::CAUTIOUS_PROTOCOL,
    )?;
    let protocol: Value = serde_json::from_str(retained::experiment::PROTOCOL)?;
    if retained::experiment::fingerprint() != PRODUCER
        || meta["source_fingerprint"] != PRODUCER
        || meta["binary_sha256"] != BINARY
        || meta["protocol"] != protocol
        || meta["original"] != row
        || meta["selection"] != task.selection
        || meta["variant"] != variant
        || meta["mode"] != mode
        || meta["arm"] != "retained-small"
        || receipt["source_fingerprint"] != PRODUCER
        || receipt["status"] != "sealed"
        || receipt["heldout_scored"] != false
        || receipt["updates"] != 512
        || receipt["endpoint_sha256"] != expected["endpoint_sha256"]
        || receipt["cache_sha256"] != expected["cache_sha256"]
    {
        return Err("content input producer or original data mismatch".into());
    }
    let model: Model = serde_json::from_slice(&fs::read(path.join("endpoint.json"))?)?;
    let parent: retained::cache::Cache =
        serde_json::from_slice(&fs::read(path.join("cache.json"))?)?;
    model.validate()?;
    parent.validate(&model)?;
    if model.objects != task.vocab
        || model.objects != 100
        || model.atoms.len() != 129
        || model.budget != 16.
        || model.family != rotary::atom::Family::Rotary
        || parent.coverage() != parent.identities.len()
        || readouts::source::content_hash(&model)?
            != receipt["content_sha256"].as_str().ok_or("input content")?
    {
        return Err("incomplete learned input family/cache".into());
    }
    Ok((model, parent, expected, task))
}

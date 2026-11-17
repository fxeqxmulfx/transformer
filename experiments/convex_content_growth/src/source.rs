//! All eight training-qualified inputs; loading never opens a score file.
use crate::{Result, experiment::Cell, model::Model};
use bilinear::{experiment as original, guard};
use serde_json::Value;
use std::{
    fs,
    path::{Path, PathBuf},
};
pub const INPUTS: &str = include_str!("../inputs.json");
pub const PRODUCER: &str = "9ee1afcef03c26a3a1e572b02852378f915a918890c15b1a09a821636e1ca8ca";
pub const BINARY: &str = "8273b98c575bf562269a761d13fbda985362ac6d56375964e894669c7f66bc37";
pub fn pin(cell: &Cell) -> Result<Value> {
    let pins: Vec<Value> = serde_json::from_str(INPUTS)?;
    pins.into_iter()
        .find(|p| {
            p["geometry"] == cell.geometry && p["variant"] == cell.variant && p["mode"] == cell.mode
        })
        .ok_or_else(|| "unlisted growth input".into())
}
pub fn path(storage: &Path, cell: &Cell) -> PathBuf {
    storage
        .join("transformer-runs/hilbert-content/original-v1")
        .join(&cell.geometry)
        .join(&cell.variant)
        .join(&cell.mode)
}
pub fn load(
    workspace: &Path,
    storage: &Path,
    cell: &Cell,
) -> Result<(Model, Value, baseline::data::Task)> {
    guard::admit(0, workspace, storage)?;
    let pin = pin(cell)?;
    let path = path(storage, cell);
    for (file, key) in [
        ("metadata.json", "metadata"),
        ("training_receipt.json", "receipt"),
        ("endpoint.json", "endpoint"),
    ] {
        if pin[key] != original::hash(&fs::read(path.join(file))?) {
            return Err("immutable growth input changed".into());
        }
    }
    let m = original::score::read(&path.join("metadata.json"))?;
    let r = original::score::read(&path.join("training_receipt.json"))?;
    let old_protocol: Value = serde_json::from_str(hilbert::experiment::PROTOCOL)?;
    if hilbert::experiment::fingerprint() != PRODUCER
        || m["source_fingerprint"] != PRODUCER
        || m["binary_sha256"] != BINARY
        || m["protocol"] != old_protocol
        || m["arm"] != cell.geometry
        || m["variant"] != cell.variant
        || m["mode"] != cell.mode
        || r["source_fingerprint"] != PRODUCER
        || r["status"] != "sealed"
        || r["heldout_scored"] != false
        || r["updates"] != 512
        || r["endpoint_sha256"] != pin["endpoint"]
    {
        return Err("growth input provenance mismatch".into());
    }
    // This frozen loader validates parents and original examples, not scores.
    let (old, row, task) = hilbert::score::load(workspace, storage, &path)?;
    let model = Model::from_source(old)?;
    if model.objects() != 100 || model.budget() != 16. || model.components() != 130 {
        return Err("complete initial family changed".into());
    }
    Ok((model, row, task))
}

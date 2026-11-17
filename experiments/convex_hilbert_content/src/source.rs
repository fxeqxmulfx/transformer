//! Frozen learned parent inputs. Training never opens held-out score files.
use crate::Result;
use bilinear::{experiment as original, guard};
use rotary::model::Model;
use serde_json::Value;
use std::{fs, path::Path};
pub const INPUTS: &str = include_str!("../inputs.json");
pub const PRODUCER: &str = "377b1055938f8f7b8c687409c730d2617eedd52420663d0c38dbfd6db67cdbea";
pub const BINARY: &str = "c9ba04ee528025f7cfd4ec38acf658317d1b5ca32b55f889c5c14a3953c6a42d";
pub fn pin(variant: &str, mode: &str) -> Result<Value> {
    let pins: Vec<Value> = serde_json::from_str(INPUTS)?;
    pins.into_iter()
        .find(|p| p["variant"] == variant && p["mode"] == mode)
        .ok_or_else(|| "unlisted learned parent".into())
}
pub fn load(
    workspace: &Path,
    storage: &Path,
    variant: &str,
    mode: &str,
) -> Result<(Model, Value, baseline::data::Task)> {
    guard::admit(0, workspace, storage)?;
    let pin = pin(variant, mode)?;
    let path = storage
        .join("transformer-runs/content-revisit/original-v1/control")
        .join(variant)
        .join(mode);
    guard::admit(
        fs::metadata(path.join("endpoint.json"))?
            .len()
            .checked_mul(8)
            .ok_or("input reservation")?,
        workspace,
        storage,
    )?;
    for (file, key) in [
        ("metadata.json", "metadata_sha256"),
        ("training_receipt.json", "receipt_sha256"),
        ("endpoint.json", "endpoint_sha256"),
    ] {
        if pin[key] != original::hash(&fs::read(path.join(file))?) {
            return Err("immutable Hilbert input changed".into());
        }
    }
    let meta = original::score::read(&path.join("metadata.json"))?;
    let r = original::score::read(&path.join("training_receipt.json"))?;
    let (_, row, task) = original::case_in(
        readouts::source::CASE,
        rotary::experiment::CAUTIOUS_PROTOCOL,
    )?;
    let p: Value = serde_json::from_str(revisit::experiment::PROTOCOL)?;
    if revisit::experiment::fingerprint() != PRODUCER
        || meta["source_fingerprint"] != PRODUCER
        || meta["binary_sha256"] != BINARY
        || meta["protocol"] != p
        || meta["original"] != row
        || meta["selection"] != task.selection
        || meta["arm"] != "control"
        || meta["variant"] != variant
        || meta["mode"] != mode
        || r["source_fingerprint"] != PRODUCER
        || r["status"] != "sealed"
        || r["heldout_scored"] != false
        || r["updates"] != 512
        || r["endpoint_sha256"] != pin["endpoint_sha256"]
    {
        return Err("learned parent provenance mismatch".into());
    }
    let parent: Model = serde_json::from_slice(&fs::read(path.join("endpoint.json"))?)?;
    parent.validate()?;
    if parent.objects != task.vocab
        || parent.objects != 100
        || parent.atoms.len() != 129
        || parent.budget != 16.
        || parent.family != rotary::atom::Family::Rotary
        || readouts::source::content_hash(&parent)? != r["content_sha256"]
    {
        return Err("original complete parent changed".into());
    }
    Ok((parent, pin, task))
}

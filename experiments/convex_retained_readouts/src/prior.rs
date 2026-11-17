//! Immutable already-scored instantaneous-large comparison; no endpoint selection.
use crate::{Result, experiment};
use bilinear::{experiment as original, guard};
use serde_json::{Value, json};
use std::{fs, path::Path};
pub const INPUTS: &str = include_str!("../baseline_inputs.json");
pub const FINGERPRINT: &str = "a4aa537ad5e8bad152e79689913078a0865394c17cd98eb63f7dfbac23168c39";
pub const BINARY: &str = "31707f5076fa26d191d47cc0eee19f7e237eddb88c880f2eb7e44c69041f3b23";
pub fn run(workspace: &Path, storage: &Path, cohort: &Path, output: &Path) -> Result<()> {
    guard::admit(0, workspace, storage)?;
    let pins: Vec<Value> = serde_json::from_str(INPUTS)?;
    let mut results = Vec::new();
    for pin in pins {
        let variant = pin["variant"].as_str().ok_or("variant")?;
        let mode = pin["mode"].as_str().ok_or("mode")?;
        let (p, _) = joint::experiment::settings(variant, mode)?;
        let path = joint::experiment::cell(cohort, variant, mode);
        for (file, key) in [
            ("metadata.json", "metadata_sha256"),
            ("training_receipt.json", "receipt_sha256"),
            ("endpoint.json", "endpoint_sha256"),
            ("scores.json", "score_sha256"),
            ("replay.json", "replay_sha256"),
        ] {
            if pin[key] != original::hash(&fs::read(path.join(file))?) {
                return Err("pinned prior changed".into());
            }
        }
        let meta = original::score::read(&path.join("metadata.json"))?;
        let receipt = original::score::read(&path.join("training_receipt.json"))?;
        let score = original::score::read(&path.join("scores.json"))?;
        let replay = original::score::read(&path.join("replay.json"))?;
        if joint::experiment::fingerprint() != FINGERPRINT
            || meta["source_fingerprint"] != FINGERPRINT
            || meta["binary_sha256"] != BINARY
            || meta["protocol"] != p
            || receipt["status"] != "sealed"
            || score["endpoint_sha256"] != pin["endpoint_sha256"]
            || replay["status"] != "identical"
        {
            return Err("prior producer identity changed".into());
        }
        results.push(json!({"pin":pin,"scores":score}));
    }
    original::write_json(
        output,
        &json!({"source_fingerprint":experiment::fingerprint(),
        "comparison":"previously scored instantaneous-large, not selected or retrained", "cells":results}),
    )
}

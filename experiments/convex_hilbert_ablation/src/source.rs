//! Immutable endpoint/score pins, independent of the new diagnostic executable.
use crate::Result;
use bilinear::{experiment as original, guard};
use serde_json::Value;
use std::{fs, path::Path};
pub const INPUTS: &str = include_str!("../inputs.json");
pub const PRODUCER: &str = "9ee1afcef03c26a3a1e572b02852378f915a918890c15b1a09a821636e1ca8ca";
pub const BINARY: &str = "8273b98c575bf562269a761d13fbda985362ac6d56375964e894669c7f66bc37";
pub fn verify(workspace: &Path, storage: &Path, cohort: &Path) -> Result<Vec<Value>> {
    guard::admit(64 * 1024 * 1024, workspace, storage)?;
    let pins: Vec<Value> = serde_json::from_str(INPUTS)?;
    let protocol: Value = serde_json::from_str(hilbert::experiment::PROTOCOL)?;
    if pins.len() != 8 || hilbert::experiment::fingerprint() != PRODUCER {
        return Err("ablation input producer changed".into());
    }
    for arm in hilbert::experiment::ARMS {
        for variant in hilbert::experiment::VARIANTS {
            for mode in hilbert::experiment::MODES {
                let p = pins
                    .iter()
                    .find(|p| p["arm"] == arm && p["variant"] == variant && p["mode"] == mode)
                    .ok_or("incomplete ablation cohort")?;
                let path = hilbert::experiment::cell(cohort, arm, variant, mode);
                for (file, key) in [
                    ("metadata.json", "metadata"),
                    ("training_receipt.json", "receipt"),
                    ("endpoint.json", "endpoint"),
                    ("scores.json", "scores"),
                ] {
                    if original::hash(&fs::read(path.join(file))?) != p[key] {
                        return Err("immutable ablation input hash changed".into());
                    }
                }
                let meta = original::score::read(&path.join("metadata.json"))?;
                let r = original::score::read(&path.join("training_receipt.json"))?;
                let s = original::score::read(&path.join("scores.json"))?;
                if meta["source_fingerprint"] != PRODUCER
                    || meta["binary_sha256"] != BINARY
                    || meta["protocol"] != protocol
                    || meta["arm"] != arm
                    || meta["variant"] != variant
                    || meta["mode"] != mode
                    || r["status"] != "sealed"
                    || r["heldout_scored"] != false
                    || r["source_fingerprint"] != PRODUCER
                    || r["endpoint_sha256"] != p["endpoint"]
                    || s["source_fingerprint"] != PRODUCER
                    || s["endpoint_sha256"] != p["endpoint"]
                {
                    return Err("ablation parent provenance mismatch".into());
                }
            }
        }
    }
    Ok(pins)
}

//! Immutable original examples, learned parameters, and complete raw-score pins.
use crate::Result;
use bilinear::{experiment as original, guard};
use growth::{
    experiment::{self, Cell},
    model::Model,
};
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
};
pub const INPUTS: &str = include_str!("../inputs.json");
pub const PRODUCER: &str = "24ac015c45ea111d4edb070699bdb170f443eaa02e5b7b959e171926f991d727";
pub const BINARY: &str = "c23dd1d88034d5cf7ba5240d58e1a8232effe8a32b7c8ca7e6decf78c8fcffb5";
pub fn cell(p: &Value) -> Result<Cell> {
    let kind = p["kind"].as_str().ok_or("kind")?;
    Cell::new(
        if kind == "initial" { "continue" } else { kind },
        p["geometry"].as_str().ok_or("geometry")?,
        p["variant"].as_str().ok_or("variant")?,
        p["mode"].as_str().ok_or("mode")?,
    )
}
pub fn label(p: &Value) -> Result<String> {
    let c = cell(p)?;
    Ok(format!(
        "{}-{}-{}-{}",
        p["kind"].as_str().ok_or("kind")?,
        c.geometry,
        c.variant,
        c.mode
    ))
}
pub fn path(storage: &Path, cohort: &Path, p: &Value) -> Result<PathBuf> {
    let c = cell(p)?;
    Ok(if p["kind"] == "initial" {
        growth::source::path(storage, &c)
    } else {
        c.path(cohort)
    })
}
pub fn verify(workspace: &Path, storage: &Path, cohort: &Path) -> Result<Vec<Value>> {
    guard::admit(256 * 1024 * 1024, workspace, storage)?;
    let pins: Vec<Value> = serde_json::from_str(INPUTS)?;
    if pins.len() != 24
        || experiment::fingerprint() != PRODUCER
        || hilbert::experiment::fingerprint() != growth::source::PRODUCER
    {
        return Err("binding input producer changed".into());
    }
    for kind in ["initial", "continue", "grow"] {
        for g in ["old", "hilbert"] {
            for v in ["cautious", "refined"] {
                for mode in ["mini", "forward"] {
                    let p = pins
                        .iter()
                        .find(|p| {
                            p["kind"] == kind
                                && p["geometry"] == g
                                && p["variant"] == v
                                && p["mode"] == mode
                        })
                        .ok_or("incomplete binding cohort")?;
                    let c = cell(p)?;
                    let path = path(storage, cohort, p)?;
                    let endpoint = if kind == "initial" {
                        "endpoint.json"
                    } else {
                        "stage-8/endpoint.json"
                    };
                    for (file, key) in [
                        ("metadata.json", "metadata"),
                        ("training_receipt.json", "receipt"),
                        (endpoint, "endpoint"),
                        ("scores.json", "scores"),
                        ("replay.json", "replay"),
                    ] {
                        if original::hash(&fs::read(path.join(file))?) != p[key] {
                            return Err("immutable binding input hash changed".into());
                        }
                    }
                    let m = original::score::read(&path.join("metadata.json"))?;
                    let r = original::score::read(&path.join("training_receipt.json"))?;
                    let s = original::score::read(&path.join("scores.json"))?;
                    let replay = original::score::read(&path.join("replay.json"))?;
                    let producer = if kind == "initial" {
                        growth::source::PRODUCER
                    } else {
                        PRODUCER
                    };
                    let binary = if kind == "initial" {
                        growth::source::BINARY
                    } else {
                        BINARY
                    };
                    let protocol: Value = serde_json::from_str(if kind == "initial" {
                        hilbert::experiment::PROTOCOL
                    } else {
                        experiment::PROTOCOL
                    })?;
                    if m["source_fingerprint"] != producer
                        || m["binary_sha256"] != binary
                        || m["protocol"] != protocol
                        || r["source_fingerprint"] != producer
                        || r["status"] != "sealed"
                        || r["heldout_scored"] != false
                        || s["source_fingerprint"] != producer
                        || replay["status"] != "identical"
                        || replay["score_sha256"] != p["scores"]
                    {
                        return Err("binding input provenance mismatch".into());
                    }
                    if kind == "initial" {
                        if m["arm"] != g
                            || m["variant"] != v
                            || m["mode"] != mode
                            || r["updates"] != 512
                            || r["endpoint_sha256"] != p["endpoint"]
                            || s["endpoint_sha256"] != p["endpoint"]
                        {
                            return Err("initial binding input changed".into());
                        }
                    } else {
                        if m["cell"] != json!(c)
                            || m["input"] != growth::source::pin(&c)?
                            || r["cell"] != json!(c)
                            || s["cell"] != json!(c)
                            || r["updates"] != 4096
                            || r["stage_receipts"].as_array().map(Vec::len) != Some(8)
                        {
                            return Err("final binding input changed".into());
                        }
                        for n in 1..=8 {
                            let stage = experiment::stage(&path, n);
                            let bytes = fs::read(stage.join("training_receipt.json"))?;
                            let sr: Value = serde_json::from_slice(&bytes)?;
                            if r["stage_receipts"][n - 1] != original::hash(&bytes)
                                || sr["endpoint_sha256"]
                                    != original::hash(&fs::read(stage.join("endpoint.json"))?)
                                || sr["source_fingerprint"] != producer
                                || sr["stage"] != n
                                || sr["status"] != "sealed"
                                || sr["heldout_scored"] != false
                                || sr["updates"] != 512
                            {
                                return Err("binding input stage changed".into());
                            }
                            if n == 8
                                && (r["final_sha256"] != sr["final_sha256"]
                                    || s["checkpoints"]["8"]["model_sha256"] != sr["final_sha256"])
                            {
                                return Err("binding endpoint identity changed".into());
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(pins)
}
pub fn load(
    workspace: &Path,
    storage: &Path,
    cohort: &Path,
    p: &Value,
) -> Result<(Model, Value, baseline::data::Task, Value)> {
    let c = cell(p)?;
    let path = path(storage, cohort, p)?;
    let (mut model, row, task) = if p["kind"] == "initial" {
        growth::source::load(workspace, storage, &c)?
    } else {
        growth::seal::initial(workspace, storage, cohort, &c)?
    };
    let scores = original::score::read(&path.join("scores.json"))?;
    let reference = if p["kind"] == "initial" {
        scores
    } else {
        let protocol = experiment::protocol()?;
        for n in 1..=8 {
            let active = experiment::prepare(&mut model, &c, n, &protocol)?;
            model = growth::seal::transition(&model, &experiment::stage(&path, n), active)?;
        }
        scores["checkpoints"]["8"].clone()
    };
    model.validate()?;
    Ok((model, row, task, reference))
}

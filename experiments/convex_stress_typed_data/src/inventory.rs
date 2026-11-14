//! Freeze source identities, all data fingerprints and historical comparisons;
//! this command neither constructs a model nor invents an accuracy measurement.
use crate::*;
use serde_json::{Value, json};
use std::{fs, path::Path};
pub fn read_pinned(path: &Path, expected: &str) -> Result<Value> {
    let bytes = fs::read(path)?;
    if hash(&bytes) != expected {
        return Err("frozen input bytes differ".into());
    }
    Ok(serde_json::from_slice(&bytes)?)
}
pub fn prepare(history: &Path, baseline: &Path, output: &Path) -> Result<()> {
    if output.exists() {
        return Err("fresh typed inventory output required".into());
    }
    let p: Value = serde_json::from_str(PROTOCOL)?;
    let history = read_pinned(history, p["history_sha256"].as_str().unwrap())?;
    let baseline = read_pinned(baseline, p["baseline324_sha256"].as_str().unwrap())?;
    let mut rows = vec![];
    for case in CASES {
        let data = dataset::build(case)?;
        let old: Vec<_> = history["runs"]
            .as_array()
            .ok_or("missing historical runs")?
            .iter()
            .filter(|r| r["case"] == case)
            .collect();
        if old.len() != 1 {
            return Err("missing or duplicate historical typed case".into());
        }
        rows.push(json!({"case":case,"data":data.metadata(),"historical_gpt":old[0],"new_model_results":null}));
    }
    sequences::runner::artifacts::write(
        output,
        &json!({"protocol":p,"protocol_sha256":hash(PROTOCOL.as_bytes()),
        "cases":rows,"baseline324":baseline,"new_model_evaluations":0,"historical_case_fingerprints_available":false}),
    )
}

//! Training-only inspection of an externally hash-bound complete parent.
//! The fixed parent/context chain leads to original records and its complete
//! source head. No metrics, generator or heldout record is consulted.
use crate::*;
use serde_json::{Value, json};
use std::{fs, io::Write, path::Path};

pub fn extract(parent: &Value, role: &str, parent_sha256: &str) -> Result<Input> {
    if !["prior", "supplement"].contains(&role) {
        return Err("invalid parent role".into());
    }
    let mut node = parent
        .get(role)
        .filter(|n| !n.is_null())
        .ok_or("parent role absent")?;
    for _ in 0..16 {
        if node.get("source").is_some() && node.get("records").is_some() {
            let source: kernel::Model = serde_json::from_value(node["source"].clone())?;
            let records: Vec<Record> = serde_json::from_value(node["records"].clone())?;
            return Input::from_source(&source, records, parent_sha256.into());
        }
        node = node
            .get("parent")
            .or_else(|| node.get("context"))
            .ok_or("complete parent source chain missing")?;
    }
    Err("parent source chain exceeds16 links".into())
}
fn write<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    serde_json::to_writer(&mut file, value)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(())
}
pub fn run(
    parent_path: &Path,
    expected: &str,
    role: &str,
    mode: training::Mode,
    root: &Path,
) -> Result<()> {
    if root.exists() {
        return Err("fresh inspection directory required".into());
    }
    let bytes = fs::metadata(parent_path)?.len();
    let parse_bound = bytes
        .checked_mul(32)
        .and_then(|x| x.checked_add(1 << 30))
        .ok_or("parent parse forecast overflow")?;
    if bytes > 1 << 30 || parse_bound > 56 * 1024u64.pow(3) {
        return Err("parent JSON parse allowance refused".into());
    }
    let input = {
        let bytes = fs::read(parent_path)?;
        if hash(&bytes) != expected {
            return Err("complete parent hash differs from frozen reference".into());
        }
        let parent: Value = serde_json::from_slice(&bytes)?;
        extract(&parent, role, expected)?
    };
    let policy = training::policies()?
        .into_iter()
        .find(|p| p.mode == mode)
        .ok_or("mode absent from protocol")?;
    let forecast = limits::forecast(&input, &policy)?;
    if forecast.buffer_bytes + parse_bound > 56 * 1024u64.pow(3) {
        return Err("parent parse plus training buffer allowance refused".into());
    }
    fs::create_dir(root)?;
    write(&root.join("input.json"), &input)?;
    write(
        &root.join("forecast.json"),
        &json!({"parent_bytes":bytes,"parent_parse_buffer_forecast":parse_bound,
        "role":role,"policy":policy,"component":forecast,"heldout_scores":0}),
    )?;
    let outcome = training::run(input.clone(), policy)?;
    write(&root.join("outcome.json"), &outcome)?;
    let loaded: training::Outcome = serde_json::from_slice(&fs::read(root.join("outcome.json"))?)?;
    training::replay(&input, &loaded)?;
    write(
        &root.join("summary.json"),
        &json!({"mode":mode,"parent_sha256":expected,"role":role,
        "input_sha256":outcome.input_sha256,"initial_model_sha256":outcome.initial_model_sha256,
        "model_sha256":digest(&outcome.model)?,"records":input.records.len(),"blocks":input.blocks,
        "vocabulary":input.vocabulary,"initial_ce":outcome.initial_ce,"final_ce":outcome.final_ce,
        "initialization":outcome.initialization,"work":outcome.work,"bounds":outcome.bounds,
        "state_replay_exact":true,"original_heldout_evaluations":0,
        "native_target_realizability_certified":false}),
    )
}

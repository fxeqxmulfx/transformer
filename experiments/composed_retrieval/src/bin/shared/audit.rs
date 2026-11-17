//! Invoke the independently built raw decoder and check its catalog identity.
use composed_retrieval::Result;
use serde_json::Value;
use std::{fs, path::Path, process::Command};

pub fn audit(workspace: &Path, storage: &Path, root: &Path) -> Result<Value> {
    let out = Command::new(root.join("original/target/release/audit-raw"))
        .args([workspace, storage, root])
        .output()?;
    if !out.status.success() {
        return Err(format!("raw audit failed: {}", String::from_utf8_lossy(&out.stderr)).into());
    }
    let a: Value = serde_json::from_slice(&out.stdout)?;
    if a["producer"] != fs::read_to_string(root.join("producer.commit"))?.trim()
        || a["summary"]["catalog"] != 165
        || a["cases"].as_object().ok_or("case ledger")?.len() != 165
    {
        return Err("incomplete raw audit".into());
    }
    Ok(a)
}

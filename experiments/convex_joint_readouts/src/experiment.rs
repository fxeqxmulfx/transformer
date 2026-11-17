//! Frozen prospective protocol and producer identity; no score-driven selection.
use crate::{Result, update::Config};
use bilinear::experiment as original;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
pub const PROTOCOL: &str = include_str!("../protocol.json");
pub const VARIANTS: [&str; 2] = ["cautious", "refined"];
pub const MODES: [&str; 2] = ["mini", "forward"];
pub fn fingerprint() -> String {
    let mut h = Sha256::new();
    for bytes in [
        include_bytes!("lib.rs").as_slice(),
        include_bytes!("capacity.rs"),
        include_bytes!("coefficient.rs"),
        include_bytes!("diagnostic.rs"),
        include_bytes!("update.rs"),
        include_bytes!("experiment.rs"),
        include_bytes!("diagnose.rs"),
        include_bytes!("train.rs"),
        include_bytes!("score.rs"),
        include_bytes!("main.rs"),
        include_bytes!("../Cargo.toml"),
        include_bytes!("../Cargo.lock"),
        PROTOCOL.as_bytes(),
        optimality::experiment::fingerprint().as_bytes(),
    ] {
        h.update((bytes.len() as u64).to_le_bytes());
        h.update(bytes);
    }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}
pub fn settings(variant: &str, mode: &str) -> Result<(Value, Config)> {
    if !VARIANTS.contains(&variant) || !MODES.contains(&mode) {
        return Err("cell outside frozen cohort".into());
    }
    let p: Value = serde_json::from_str(PROTOCOL)?;
    let c = Config {
        mode: original::mode(mode)?,
        query_step: p["query_step"].as_f64().ok_or("query step")?,
        line_iterations: p["line_iterations"].as_u64().ok_or("line iterations")? as usize,
        step_ceiling: p["step_ceiling"].as_f64().ok_or("step ceiling")?,
    };
    Ok((p, c))
}
pub fn cell(cohort: &Path, variant: &str, mode: &str) -> PathBuf {
    cohort.join(variant).join(mode)
}

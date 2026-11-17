//! Prospective factorial controls and producer identity, fixed before training.
use crate::Result;
use bilinear::experiment as original;
use joint::update::Config;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
pub const PROTOCOL: &str = include_str!("../protocol.json");
pub const ARMS: [&str; 3] = ["instantaneous-small", "retained-large", "retained-small"];
pub use joint::experiment::{MODES, VARIANTS};
pub fn fingerprint() -> String {
    let mut h = Sha256::new();
    for bytes in [
        include_bytes!("lib.rs").as_slice(),
        include_bytes!("cache.rs"),
        include_bytes!("coefficient.rs"),
        include_bytes!("diagnostic.rs"),
        include_bytes!("experiment.rs"),
        include_bytes!("index.rs"),
        include_bytes!("line.rs"),
        include_bytes!("prior.rs"),
        include_bytes!("score.rs"),
        include_bytes!("train.rs"),
        include_bytes!("update.rs"),
        include_bytes!("main.rs"),
        include_bytes!("../Cargo.toml"),
        include_bytes!("../Cargo.lock"),
        include_bytes!("../baseline_inputs.json"),
        PROTOCOL.as_bytes(),
        joint::experiment::fingerprint().as_bytes(),
    ] {
        h.update((bytes.len() as u64).to_le_bytes());
        h.update(bytes);
    }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}
pub fn settings(arm: &str, variant: &str, mode: &str) -> Result<(Value, Config, bool)> {
    if !ARMS.contains(&arm) || !VARIANTS.contains(&variant) || !MODES.contains(&mode) {
        return Err("cell outside frozen retained cohort".into());
    }
    let p: Value = serde_json::from_str(PROTOCOL)?;
    let a = p["arms"]
        .as_array()
        .ok_or("arms")?
        .iter()
        .find(|a| a["name"] == arm)
        .ok_or("arm")?;
    let c = Config {
        mode: original::mode(mode)?,
        query_step: p["query_step"].as_f64().ok_or("query step")?,
        line_iterations: p["line_iterations"].as_u64().ok_or("line iterations")? as usize,
        step_ceiling: a["step_ceiling"].as_f64().ok_or("step ceiling")?,
    };
    let retained = a["retained"].as_bool().ok_or("retained")?;
    Ok((p, c, retained))
}
pub fn cell(cohort: &Path, arm: &str, variant: &str, mode: &str) -> PathBuf {
    cohort.join(arm).join(variant).join(mode)
}

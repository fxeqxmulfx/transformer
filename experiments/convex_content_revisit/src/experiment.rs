//! Fixed matched control and content-revisit protocol, independent of scores.
use crate::{Result, content};
use bilinear::experiment as original;
use joint::update::Config;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
pub const PROTOCOL: &str = include_str!("../protocol.json");
pub const ARMS: [&str; 2] = ["control", "revisit"];
pub use retained::experiment::{MODES, VARIANTS};
pub fn fingerprint() -> String {
    let mut h = Sha256::new();
    for b in [
        include_bytes!("lib.rs").as_slice(),
        include_bytes!("cache.rs"),
        include_bytes!("content.rs"),
        include_bytes!("diagnostic.rs"),
        include_bytes!("experiment.rs"),
        include_bytes!("history.rs"),
        include_bytes!("score.rs"),
        include_bytes!("source.rs"),
        include_bytes!("train.rs"),
        include_bytes!("update.rs"),
        include_bytes!("main.rs"),
        include_bytes!("../Cargo.toml"),
        include_bytes!("../Cargo.lock"),
        include_bytes!("../inputs.json"),
        PROTOCOL.as_bytes(),
        retained::experiment::fingerprint().as_bytes(),
    ] {
        h.update((b.len() as u64).to_le_bytes());
        h.update(b);
    }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}
pub fn settings(
    arm: &str,
    variant: &str,
    mode: &str,
) -> Result<(Value, Config, content::Config, bool)> {
    if !ARMS.contains(&arm) || !VARIANTS.contains(&variant) || !MODES.contains(&mode) {
        return Err("cell outside content cohort".into());
    }
    let p: Value = serde_json::from_str(PROTOCOL)?;
    let c = Config {
        mode: original::mode(mode)?,
        query_step: p["query_step"].as_f64().ok_or("query step")?,
        line_iterations: p["line_iterations"].as_u64().ok_or("line iterations")? as usize,
        step_ceiling: p["step_ceiling"].as_f64().ok_or("step ceiling")?,
    };
    let content = content::Config {
        radius: p["content_radius"].as_f64().ok_or("content radius")?,
        frequency_radius: p["frequency_radius"].as_f64().ok_or("frequency radius")?,
    };
    Ok((p, c, content, arm == "revisit"))
}
pub fn cell(cohort: &Path, arm: &str, variant: &str, mode: &str) -> PathBuf {
    cohort.join(arm).join(variant).join(mode)
}

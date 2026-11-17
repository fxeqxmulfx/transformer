//! Prospective geometry comparison, frozen independently of held-out outcomes.
use crate::{Result, block::Block, geometry::Geometry, model::Model};
use bilinear::{experiment as original, train::Mode};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
pub const PROTOCOL: &str = include_str!("../protocol.json");
pub const ARMS: [&str; 2] = ["old", "hilbert"];
pub const VARIANTS: [&str; 2] = ["cautious", "refined"];
pub const MODES: [&str; 2] = ["mini", "forward"];
pub fn fingerprint() -> String {
    let mut h = Sha256::new();
    for b in [
        include_bytes!("lib.rs").as_slice(),
        include_bytes!("block.rs"),
        include_bytes!("cache.rs"),
        include_bytes!("diagnostic.rs"),
        include_bytes!("estimate.rs"),
        include_bytes!("experiment.rs"),
        include_bytes!("geometry.rs"),
        include_bytes!("history.rs"),
        include_bytes!("model.rs"),
        include_bytes!("score.rs"),
        include_bytes!("source.rs"),
        include_bytes!("train.rs"),
        include_bytes!("main.rs"),
        include_bytes!("../Cargo.toml"),
        include_bytes!("../Cargo.lock"),
        include_bytes!("../inputs.json"),
        PROTOCOL.as_bytes(),
        revisit::experiment::fingerprint().as_bytes(),
    ] {
        h.update((b.len() as u64).to_le_bytes());
        h.update(b);
    }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}
pub fn settings(arm: &str, variant: &str, mode: &str) -> Result<(Value, Geometry, Mode)> {
    if !ARMS.contains(&arm) || !VARIANTS.contains(&variant) || !MODES.contains(&mode) {
        return Err("cell outside geometry cohort".into());
    }
    Ok((
        serde_json::from_str(PROTOCOL)?,
        Geometry::parse(arm)?,
        original::mode(mode)?,
    ))
}
pub fn initial(parent: rotary::model::Model, geometry: Geometry, p: &Value) -> Result<Model> {
    let block = Block::initial(
        parent.objects,
        geometry,
        p["initialization_seed"]
            .as_u64()
            .ok_or("initial seed")?
            .try_into()?,
        p["initial_radius"].as_f64().ok_or("initial radius")?,
        p["matrix_jitter"].as_f64().ok_or("jitter")?,
    )?;
    let model = Model {
        parent,
        block,
        fraction: p["new_mass_fraction"].as_f64().ok_or("new mass")?,
    };
    model.validate()?;
    Ok(model)
}
pub fn cell(cohort: &Path, arm: &str, variant: &str, mode: &str) -> PathBuf {
    cohort.join(arm).join(variant).join(mode)
}

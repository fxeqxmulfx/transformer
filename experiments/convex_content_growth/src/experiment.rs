//! Fixed cohort, stage budgets, and recursive immutable source identity.
use crate::{Result, model::Model};
use hilbert::block::Block;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
pub const PROTOCOL: &str = include_str!("../protocol.json");
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cell {
    pub arm: String,
    pub geometry: String,
    pub variant: String,
    pub mode: String,
}
impl Cell {
    pub fn new(arm: &str, geometry: &str, variant: &str, mode: &str) -> Result<Self> {
        if !["continue", "grow"].contains(&arm) {
            return Err("unknown growth arm".into());
        }
        hilbert::experiment::settings(geometry, variant, mode)?;
        Ok(Self {
            arm: arm.into(),
            geometry: geometry.into(),
            variant: variant.into(),
            mode: mode.into(),
        })
    }
    pub fn path(&self, cohort: &Path) -> PathBuf {
        cohort
            .join(&self.arm)
            .join(&self.geometry)
            .join(&self.variant)
            .join(&self.mode)
    }
    pub fn label(&self) -> String {
        format!(
            "{}-{}-{}-{}",
            self.arm, self.geometry, self.variant, self.mode
        )
    }
}
pub fn cells() -> Vec<Cell> {
    let mut result = Vec::new();
    for a in ["continue", "grow"] {
        for g in ["old", "hilbert"] {
            for v in ["cautious", "refined"] {
                for m in ["mini", "forward"] {
                    result.push(Cell::new(a, g, v, m).expect("declared cell"));
                }
            }
        }
    }
    result
}
pub fn protocol() -> Result<Value> {
    Ok(serde_json::from_str(PROTOCOL)?)
}
pub fn stage(path: &Path, n: usize) -> PathBuf {
    path.join(format!("stage-{n}"))
}
pub fn prepare(model: &mut Model, cell: &Cell, n: usize, p: &Value) -> Result<usize> {
    if n == 0 || n > p["stages"].as_u64().ok_or("stages")? as usize {
        return Err("stage outside protocol".into());
    }
    if cell.arm == "continue" {
        return Ok(0);
    }
    let block = Block::initial(
        model.objects(),
        model.geometry()?,
        (p["initialization_seed_base"]
            .as_u64()
            .ok_or("initial seed")?
            + n as u64)
            .try_into()?,
        p["initial_radius"].as_f64().ok_or("radius")?,
        p["matrix_jitter"].as_f64().ok_or("jitter")?,
    )?;
    model.append(block, p["new_mass_fraction"].as_f64().ok_or("fraction")?)
}
pub fn fingerprint() -> String {
    let mut h = Sha256::new();
    for b in [
        include_bytes!("lib.rs").as_slice(),
        include_bytes!("diagnostic.rs"),
        include_bytes!("estimate.rs"),
        include_bytes!("experiment.rs"),
        include_bytes!("model.rs"),
        include_bytes!("phase.rs"),
        include_bytes!("query.rs"),
        include_bytes!("score.rs"),
        include_bytes!("seal.rs"),
        include_bytes!("source.rs"),
        include_bytes!("train.rs"),
        include_bytes!("main.rs"),
        include_bytes!("../Cargo.toml"),
        include_bytes!("../Cargo.lock"),
        include_bytes!("../inputs.json"),
        PROTOCOL.as_bytes(),
        hilbert::experiment::fingerprint().as_bytes(),
    ] {
        h.update((b.len() as u64).to_le_bytes());
        h.update(b);
    }
    h.finalize().iter().map(|x| format!("{x:02x}")).collect()
}

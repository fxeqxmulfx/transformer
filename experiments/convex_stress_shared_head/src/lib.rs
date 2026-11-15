//! Fixed shared observed/query features and a fresh complete convex CE head.
//! Lean audit5570b115 checks CD8, CM42 and CM51 without a Dirac-row premise.
//! CE and global L1 projection reuse convex_stress_depth/src/head.rs at
//! 3dac93c32fed41063a8618e4dc83e066da8a8af7. Features now use complete complex
//! products; initialization uses explicit ordinal separators/class moments.
pub use convex_stress_depth::head::{ce_gradient, cross_entropy, project_columns};
pub use convex_stress_shared_roles::{C, Result, kernel, records::Record};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
pub mod features;
pub mod initialize;
pub mod inspection;
pub mod limits;
pub mod model;
#[cfg(test)]
mod tests;
pub mod training;
pub use model::{Input, Model};

pub const PROTOCOL: &str = include_str!("../protocol.json");
pub const STATES: usize = 8;
pub const WIDTH: usize = 2 * STATES;
pub const RADIUS: f64 = 65_536.;
pub fn digest<T: Serialize>(value: &T) -> Result<String> {
    Ok(hash(&serde_json::to_vec(value)?))
}
pub fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

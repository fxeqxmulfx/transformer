//! CM55/CM51, audited before fitting in BoundarySharingAxioms.lean. Fixed input
//! descriptors tie intercepts across the complete complex head. Categorical
//! decoder discovery and constructor selection precede ordinary convex CE.
pub use convex_stress_shared_geometry::{
    C, DIRECTIONS, Input, Mode, Policy, Record, Result, ce_gradient, cross_entropy, digest, hash,
};
use serde::{Deserialize, Serialize};
pub mod features;
pub mod fit;
pub mod geometry;
pub mod inspection;
pub mod limits;
pub mod model;
pub mod partition;
pub mod projection;
#[cfg(test)]
mod tests;
pub mod training;
pub use model::Model;
pub const PROTOCOL: &str = include_str!("../protocol.json");
pub const RADIUS: f64 = 1_048_576.;
pub const RIDGES: [f64; 3] = [0.000001, 0.0001, 0.01];
pub const GAINS: [f64; 4] = [8., 32., 128., 512.];
pub const COMMONS: [f64; 2] = [8., 32.];
pub fn protocol_hash() -> Result<String> {
    digest(&serde_json::from_str::<serde_json::Value>(PROTOCOL)?)
}

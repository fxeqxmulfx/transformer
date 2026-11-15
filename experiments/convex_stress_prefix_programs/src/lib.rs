//! CM53/CM54 count and copy programs, with categorical training dictionaries.
//! Lean first: scripts/PrefixCountingGenerationAxioms.lean. Discrete discovery
//! precedes the convex affine CE slice; no global discovery-convexity claim.
pub use convex_stress_generative_programs as parent;
pub use convex_stress_parity_data as generative;
pub use parent::{
    C, Input, Observation, Result, argmax, ce, copy, data, hash, linalg, metrics, sequences,
};
use serde::{Deserialize, Serialize};
pub mod binding;
pub mod costs;
pub mod dataset;
pub mod discovery;
pub mod evaluation;
pub mod head;
#[path = "../../convex_stress_latent_programs/src/latent.rs"]
pub mod latent;
pub mod model;
pub mod previous;
pub mod program;
pub mod runner;
#[cfg(test)]
mod tests;
pub mod training;
pub const PROTOCOL: &str = include_str!("../protocol.json");
pub const STATES: usize = 256;
pub const WORK_LIMIT: u64 = 64_000_000_000;

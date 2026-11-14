//! Native CM53, reviewed at8c7b8933. Generic fixed input summaries, observed
//! dictionaries and an affine slice of the joint convex decoder domain.
//! CM52 copying has training-only priority; no target identity selects a tier.
pub use convex_stress_shared_programs as copy;
pub use copy::{
    C, Input, Observation, Result, argmax, ce, data, dataset, hash, linalg, metrics, sequences,
};
use serde::{Deserialize, Serialize};
pub mod binding;
pub mod discovery;
pub mod evaluation;
pub mod head;
pub mod latent;
pub mod model;
pub mod program;
pub mod runner;
#[cfg(test)]
mod tests;
pub mod training;
pub const PROTOCOL: &str = include_str!("../protocol.json");
pub const STATES: usize = 32;
pub const WORK_LIMIT: u64 = 16_000_000_000;

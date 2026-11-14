//! CM53 at8c7b8933, instantiated with generic input-only stack summaries.
//! Reuse the exact task324 complete heads and ordinary-CE implementation;
//! only the fixed program bank changes. Realizability remains an extra premise.
pub use convex_stress_latent_programs as parent;
pub use convex_stress_typed_data as typed;
pub use parent::{
    C, Input, Observation, Result, STATES, WORK_LIMIT, argmax, ce, data, hash, head, metrics,
    sequences,
};
use serde::{Deserialize, Serialize};
pub mod binding;
pub mod dataset;
pub mod discovery;
pub mod evaluation;
#[path = "../../convex_stress_latent_programs/src/latent.rs"]
pub mod latent;
pub mod model;
pub mod program;
pub mod runner;
#[cfg(test)]
mod tests;
pub mod training;
pub const PROTOCOL: &str = include_str!("../protocol.json");

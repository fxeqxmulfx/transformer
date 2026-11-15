//! CM53 plus CM54 at4c95deeb: accepted complete complex predictions and
//! actual own-prefix generation. Native target realizability is not certified.
//! The input-only bank learns roles and decoders from observed raw labels.
pub use convex_stress_generative_data as generative;
pub use convex_stress_nested_programs as parent;
pub use parent::parent::{copy, linalg};
pub use parent::{C, Input, Observation, Result, argmax, ce, data, hash, metrics, sequences};
use serde::{Deserialize, Serialize};
pub mod binding;
pub mod costs;
pub mod dataset;
pub mod discovery;
pub mod evaluation;
// Reviewed complete products, convex CE, projection and independent checks.
// Only the fixed source-state dimension changes from32 to128; all raw
// coefficients, construction costs and column budgets grow with that change.
pub mod head;
#[path = "../../convex_stress_latent_programs/src/latent.rs"]
pub mod latent;
#[path = "../../convex_stress_nested_programs/src/model.rs"]
pub mod model;
pub mod program;
pub mod runner;
#[cfg(test)]
mod tests;
pub mod training;
pub const PROTOCOL: &str = include_str!("../protocol.json");
pub const STATES: usize = 128;
pub const WORK_LIMIT: u64 = 64_000_000_000;

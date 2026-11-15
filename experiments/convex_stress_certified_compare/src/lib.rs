//! Separate task349 producer after CM63. Training/discovery are unchanged from
//! task347 at1371ae4f; only the complete-head numerical evaluator changes.
pub use certified::previous;
pub use convex_stress_certified_head as certified;
pub use previous::{
    Observation, Result, adapter, artifacts, dataset, discovery, fit, from_data, joint, model,
    observations, prefix, program,
};
pub const PROTOCOL: &str = include_str!("../protocol.json");
#[cfg(test)]
mod audit_tests;
pub mod disk;
pub mod evaluation;
pub mod raw;
pub mod runner;

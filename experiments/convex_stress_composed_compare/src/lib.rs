//! CM64/65/66/67/68 comparison for the fixed composition family at cbde1e2d.
//! Reuse the full tensor, model, scoring and resource machinery at f5015cec.
//! The new boundary reads complete larger probe reports directly into typed
//! structures. Production stays disabled until task358 verification is pinned.
pub use joint::Result;
pub use pairs::MAX_WORK;
#[path = "../../convex_stress_reduced_compare/src/disk.rs"]
pub mod disk;
#[path = "../../convex_stress_reduced_compare/src/evaluation.rs"]
pub mod evaluation;
#[path = "../../convex_stress_reduced_compare/src/model.rs"]
pub mod model;
#[path = "../../convex_stress_reduced_compare/src/resources.rs"]
pub mod resources;
pub mod runner;
#[path = "../../convex_stress_reduced_compare/src/tensor.rs"]
pub mod tensor;
pub const RAM_LIMIT_BYTES: u64 = 110_000_000_000;
#[cfg(test)]
mod tests;

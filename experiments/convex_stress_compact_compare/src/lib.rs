//! Separate prospective CM70 comparison. The preparation boundary retains
//! the complete task358 input-only family and original audits. No production
//! runner, heldout fitting or new accuracy result is enabled by this component.
pub use joint::{Result, Work};
#[path = "../../convex_stress_reduced_compare/src/disk.rs"]
pub mod disk;
pub mod evaluation;
mod memory;
pub mod model;
pub mod preparation;
pub mod resources;
pub mod tensor;
#[cfg(test)]
mod tests;
pub mod work;
/// User priority2026-11-16: minibatch and forward-only minibatch processing.
/// Full-batch numerical controls remain available as optional references.
pub const REQUIRED_MODES: [joint::training::Mode; 2] =
    [joint::training::Mode::Forward, joint::training::Mode::Mini];
pub const OPTIONAL_REFERENCE_MODES: [joint::training::Mode; 1] = [joint::training::Mode::Full];

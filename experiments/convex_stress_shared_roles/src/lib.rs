//! CM40 at ec5f839d: actual shared-role rows restrict the complete grouped
//! coefficient kernel. A free common offset preserves the original gauges.
//! Discovery uses observations and identical categorical token names only.
//! Unknown global shared-role realizability and readout compatibility remain
//! explicit assumptions. Convexity concerns the resulting fixed complex head.
pub use convex_stress_kernel as kernel;
pub use kernel::{C, Initialization, Model, Result, finite_system, limits};
use serde::{Deserialize, Serialize};
pub mod audit;
pub mod binding;
pub mod construct;
pub mod discovery;
pub mod prediction;
pub mod records;
#[cfg(test)]
mod tests;
pub mod training;

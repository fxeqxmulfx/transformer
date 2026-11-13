//! CM35 at ff6313e1: a declared finite output partition supplies additional
//! observed equations. Kernel completeness refers to those grouped equations;
//! the complex head and every CE derivative still use the original vocabulary.
//! Unknown-target realizability and compatibility remain explicit hypotheses.
pub use convex_stress_kernel as kernel;
pub use kernel::{C, Initialization, Model, Result, audit, limits, training};
use serde::{Deserialize, Serialize};
pub mod binding;
pub mod construct;
pub mod discovery;
mod preflight;
#[cfg(test)]
mod tests;

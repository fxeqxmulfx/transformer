//! CM31/CM32 at fbbe4617/b52ba949: observed-input refinement, fixed latent
//! Fourier blocks and feasible ordinary-CE updates in one combined head.
//! Discovery remains discrete; no arbitrary unseen-label identification claim.
pub use convex_stress_quotient as quotient;
pub use convex_stress_tagged::{C, Result, linalg, metrics};
use serde::{Deserialize, Serialize};
pub mod audit;
pub mod binding;
pub mod construct;
pub mod limits;
pub mod model;
pub mod training;
pub use construct::{Initialization, initialize};
pub use model::Model;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod training_tests;

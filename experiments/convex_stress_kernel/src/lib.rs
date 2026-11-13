//! CM34 at285dcdb6: retain a complete original observed coefficient kernel.
//! The finite-field solver/checker is reused unchanged from the frozen quotient
//! dependency. The bounded Fourier head and actual ordinary-CE paths port
//! convex_stress_refined_quotient at285dcdb6. Discovery is discrete and
//! observed-only; convexity concerns training the frozen complex feature head.
pub use convex_stress_quotient::{catalogue, finite_system};
pub use convex_stress_tagged::{C, Result, linalg, metrics};
use serde::{Deserialize, Serialize};
pub mod audit;
pub mod binding;
pub mod construct;
pub mod discovery;
pub mod limits;
pub mod model;
pub mod states;
pub mod training;
pub use construct::{Initialization, initialize};
pub use model::Model;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod training_tests;

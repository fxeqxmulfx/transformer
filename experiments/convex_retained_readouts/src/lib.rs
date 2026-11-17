//! CM105 at b87fa714: explicitly approximate complete CE coefficients.
//! Retaining old signals is an empirical estimator, not a convergence theorem.
pub use rotary::{C, Result};
pub mod cache;
pub mod coefficient;
pub mod diagnostic;
pub mod experiment;
pub mod index;
pub mod line;
pub mod prior;
pub mod score;
#[cfg(test)]
mod tests;
pub mod train;
pub mod update;

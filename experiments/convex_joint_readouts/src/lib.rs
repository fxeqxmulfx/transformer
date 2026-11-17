//! CM104 at33511122: complete weighted rows with redistributable capacity.
//! Numerical conditional optimization; no rounding or unseen-answer certificate.
pub use rotary::{C, Result};
pub mod capacity;
pub mod coefficient;
pub mod diagnose;
pub mod diagnostic;
pub mod experiment;
pub mod score;
#[cfg(test)]
mod tests;
pub mod train;
pub mod update;

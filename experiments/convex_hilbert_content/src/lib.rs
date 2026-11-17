//! CM107/108 complete learned Hilbert recurrence and empirical finite-query training.
pub use rotary::{C, Result};
pub mod block;
pub mod cache;
pub mod diagnostic;
pub mod estimate;
pub mod experiment;
pub mod geometry;
pub mod history;
pub mod model;
pub mod score;
pub mod source;
#[cfg(test)]
mod tests;
pub mod train;

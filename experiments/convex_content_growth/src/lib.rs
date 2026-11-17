//! CM107 finite mixtures and CM109 exact finite output-query reuse.
pub use hilbert::{C, Result};
pub mod diagnostic;
pub mod estimate;
pub mod experiment;
pub mod model;
pub mod phase;
pub mod query;
pub mod score;
pub mod seal;
pub mod source;
#[cfg(test)]
mod tests;
pub mod train;

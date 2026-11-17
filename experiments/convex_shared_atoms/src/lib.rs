//! CM92 complete shared complex factor domain and feasible finite loss queries.
//! Joint-domain convexity does not imply a globally optimal component search.
pub use bilinear::{C, Result};
pub mod experiment;
pub mod model;
pub mod oracle;
pub mod train;

#[cfg(test)]
mod tests;

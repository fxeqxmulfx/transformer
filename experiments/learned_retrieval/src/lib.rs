pub use bilinear::Result;
pub mod cohort;
pub mod evaluation;
pub mod gradient;
pub mod model;
pub mod train;

#[path = "../../convex_binding_diagnostic/src/counts.rs"]
pub mod binding_counts;

#[cfg(test)]
mod tests;

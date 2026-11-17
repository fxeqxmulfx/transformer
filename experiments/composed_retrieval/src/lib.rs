//! Native realization of the complete physical CM113 next-object score.
pub use baseline::Result;
#[cfg(feature = "arithmetic-cohort")]
pub mod arithmetic;
pub mod cohort;
pub mod data;
pub mod evaluation;
pub mod gradient;
pub mod model;
pub mod operator;
#[cfg(feature = "prefix-cohort")]
pub mod prefix;
mod query;
pub mod train;

#[cfg(test)]
mod tests;

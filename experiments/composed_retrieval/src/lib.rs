//! Native realization of the complete physical CM113 next-object score.
pub use baseline::Result;
#[cfg(feature = "arithmetic-cohort")]
pub mod arithmetic;
#[cfg(feature = "chain-data")]
pub mod chain_data;
#[cfg(feature = "chain-cohort")]
pub mod chains;
pub mod cohort;
pub mod data;
pub mod evaluation;
#[cfg(feature = "rewrite-cohort")]
pub mod generation;
#[cfg(feature = "generation-cohort")]
pub mod generation_cohort;
pub mod gradient;
pub mod model;
pub mod operator;
#[cfg(feature = "prefix-cohort")]
pub mod prefix;
#[cfg(any(feature = "profile-cohort", feature = "predecessor-cohort"))]
pub mod profile;
mod query;
#[cfg(feature = "output-offsets")]
pub mod radial;
#[cfg(feature = "radii-cohort")]
pub mod radii_cohort;
pub mod train;

#[cfg(test)]
mod tests;

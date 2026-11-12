//! Conventional transformer adapter for the original frozen stress datasets.
//! Reuses the Kernite kernels, layout and AdamW from convex_recall_compare at
//! da0714aa. This comparator adds no convexity or unseen-label recovery claim.
pub use convex_recall_compare::{adam, gemm, layout, mt};
pub use convex_stress_minibatch::{Result, data, hash, metrics};
pub mod checkpoint;
pub mod engine;
pub mod memory;
pub mod model;
pub mod preflight;

#[cfg(test)]
#[rustfmt::skip]
#[path = "../../grok_division/src/reference.rs"]
pub mod reference;
#[cfg(test)]
mod tests;

//! Native comparisons of the convex recall construction at 0fcc0d1b.
//!
//! The conventional transformer uses the validated Kernite CPU engine from
//! `grok_division` at 891e4525. Only its attention scratch capacity changes;
//! the copied model supports contexts longer than sixteen tokens. Its parameter
//! layout, optimizer, RNG, and GEMM are shared source modules, fingerprinted in runs.

#[rustfmt::skip]
#[path = "../../grok_division/src/adam.rs"]
pub mod adam;
#[rustfmt::skip]
#[path = "../../grok_division/src/gemm.rs"]
pub mod gemm;
#[rustfmt::skip]
#[path = "../../grok_division/src/layout.rs"]
pub mod layout;
#[rustfmt::skip]
#[path = "../../grok_division/src/mt.rs"]
pub mod mt;
#[cfg(test)]
#[rustfmt::skip]
#[path = "../../grok_division/src/reference.rs"]
pub mod reference;

pub mod convex;
pub mod data;
pub mod engine;
#[path = "transformer_model.rs"]
pub mod model;

#[cfg(test)]
mod tests;

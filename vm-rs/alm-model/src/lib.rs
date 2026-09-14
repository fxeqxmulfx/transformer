//! The compiled WASM transformer: weights, the forward pass, and generation.
//!
//! A port of `transformer_vm/model/` from the Percepta `transformer-vm`
//! release.  Everything is `f64` and nothing is a tensor: generation is one
//! token at a time, so the linear algebra is four matrix-vector products a
//! layer, and the attention is a convex-hull query rather than an inner
//! product at all (`alm_hull`).

pub mod cache;
pub mod linear;
pub mod model;
pub mod weights;

pub use cache::{CacheKind, KvCache};
pub use linear::{Dense, SparseHead};
pub use model::{Alm, Scratch, Timings};
pub use weights::{RawModel, Shapes};

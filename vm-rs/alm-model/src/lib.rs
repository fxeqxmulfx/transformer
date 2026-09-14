//! The compiled WASM transformer: weights, the forward pass, and generation.
//!
//! A port of `transformer_vm/model/` from the Percepta `transformer-vm`
//! release.  The tensor work runs on burn, pinned to the `ndarray` backend
//! with `f64` elements; the attention does not, because it is a convex-hull
//! query rather than a tensor operation (`alm_hull`).

pub mod cache;
pub mod model;
pub mod weights;

pub use cache::{CacheKind, KvCache};
pub use model::Alm;
pub use weights::{RawModel, Shapes};

/// The backend every entry point uses: `ndarray`, with `f64` elements.
pub type Backend = burn::backend::ndarray::NdArray<f64, i64, i8>;

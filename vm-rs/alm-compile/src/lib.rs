//! The compiler: the computation graph, the schedule, and the weights.
//!
//! A port of the Python side of the Percepta release — `graph/core.py`,
//! `wasm/interpreter.py` and `model/weights.py` — whose purpose is to let the
//! whole pipeline run and be verified end to end from Rust.  The check that
//! keeps the port honest is byte identity: the `model.bin` this builds must be
//! the file the Python builder writes.

pub mod decoder;
pub mod dump;
pub mod emit;
pub mod expr;
pub mod graph;
pub mod interpreter;
pub mod isa;
pub mod lower;
pub mod naming;
pub mod plan;
pub mod reference;
pub mod scheduler;
pub mod slots;
pub mod weights;

/// A path inside the vendored release, resolved against this crate rather than
/// against the working directory.  `cargo test` runs a unit test from the
/// package root, so a plain `../transformer-vm/...` silently misses and every
/// check against the released artefacts skips instead of running.
#[cfg(test)]
pub(crate) fn vendored(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../transformer-vm").join(rel)
}

pub use expr::{DimId, Expr};
pub use graph::{Graph, LookUp, TieBreak};

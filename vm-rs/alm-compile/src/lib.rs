//! The compiler: the computation graph, the schedule, and the weights.
//!
//! A port of the Python side of the Percepta release — `graph/core.py`,
//! `wasm/interpreter.py` and `model/weights.py` — whose purpose is to let the
//! whole pipeline run and be verified end to end from Rust.  The check that
//! keeps the port honest is byte identity: the `model.bin` this builds must be
//! the file the Python builder writes.

pub mod dump;
pub mod expr;
pub mod graph;
pub mod interpreter;
pub mod isa;
pub mod naming;
pub mod plan;
pub mod slots;
pub mod weights;

pub use expr::{DimId, Expr};
pub use graph::{Graph, LookUp, TieBreak};

//! Native data-only port of the frozen synthetic TypedDyck source (task318).
//! Answer rules are confined to this dataset adapter; no model is implemented.
pub use convex_stress_sequences as sequences;
pub use sequences::{Random, Result, data, hash};
pub mod dataset;
mod generator;
pub mod inventory;
#[cfg(test)]
mod tests;
#[path = "../../convex_stress_sequences/src/wire.rs"]
mod wire;
pub const PROTOCOL: &str = include_str!("../protocol.json");
pub const CASES: [&str; 2] = ["suite-dyck-2", "suite-dyck-3"];

//! Read-only independent readback for the frozen1371ae4f producer. This crate
//! is not a dependency of the producer and cannot affect its predictions.
pub use convex_stress_streaming_counts::{
    PROTOCOL, Result, adapter, artifacts, dataset, evaluation, from_data, joint, model,
    observations, program, runner,
};
pub mod raw;
pub mod run;
#[cfg(test)]
mod tests;

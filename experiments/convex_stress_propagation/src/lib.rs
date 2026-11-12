//! Native CM19 after Lean 0807609b, `DecodedPropagation/{Defs,Basic,Identification}`.
//! Arbitrary fixed decoder fibers, all anchor branches and all bank members.
//! The target-family assumption is explicit: training fit cannot establish it.
//! This is a checked native implementation, not extracted or verified Rust code.
mod algorithm;
mod check;
mod preflight;
mod types;

pub use algorithm::propagate;
pub use check::{Verified, verify};
pub use preflight::preflight;
pub use types::*;

pub type Result<T> = std::result::Result<T, String>;
pub const RUN_MEMORY_BYTES: u64 = 64 * 1024 * 1024 * 1024;

#[cfg(test)]
mod tests;

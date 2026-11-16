//! CM70 storage after Lean cc6931a4. Shared rows compare binary64 words,
//! and arbitrary nonuniform rows retain a complete explicit fallback.
//! Logical arithmetic remains that of the complete reduced model. The
//! whole-process admission and original-cohort integration remain separate.
pub use legacy::{Record, Result, Shape, Work, ce};
pub mod head;
mod memory;
mod parameters;
pub use parameters::Parameters;
pub mod projection;
pub mod rows;
#[cfg(test)]
mod tests;
pub mod training;

//! CM70 storage after Lean cc6931a4. Shared rows compare binary64 words,
//! and arbitrary nonuniform rows retain a complete explicit fallback.
//! Logical arithmetic remains that of the complete reduced model. The
//! optimizer and whole-process admission are not yet integrated.
pub use legacy::{Result, Shape, Work};
pub mod head;
mod parameters;
pub use parameters::Parameters;
pub mod rows;
#[cfg(test)]
mod tests;

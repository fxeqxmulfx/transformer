//! CM64 arbitrary fixed causal programs and CM70 complete joint coefficients.
//! Lean contract rechecked at2e9c1bb9 before this native extension. Features are
//! an explicit input prior, not a consequence of a spectral bound. No claim of
//! target realizability or unconditional identification of unseen labels.
pub use joint::{Result, Work};
pub mod features;
pub mod input;
pub mod model;
pub mod partition;
pub mod reference;
#[cfg(test)]
mod tests;

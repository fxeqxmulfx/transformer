//! CM70 storage after Lean cc6931a4. Shared rows compare binary64 words,
//! and arbitrary nonuniform rows retain a complete explicit fallback.
//! This foundation does not yet replace the complete optimizer or head.
pub use legacy::Result;
pub mod rows;
#[cfg(test)]
mod tests;

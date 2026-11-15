//! Complete CM64 comparison: preserve every tensor entry and stop on disk
//! admission failures. Numerical/model certificates retain their exact scope.
pub use joint::Result;
pub mod disk;
pub mod evaluation;
pub mod model;
pub mod resources;
pub mod runner;
pub mod tensor;
pub const RAM_LIMIT_BYTES: u64 = 110_000_000_000;
#[cfg(test)]
mod tests;

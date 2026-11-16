//! CM65/66/67 comparison adapter, ported from bounded_compare at ca6a4977.
//! All free coordinates remain explicit. The complete complex head is rebuilt
//! from a bound dictionary and bit-exact reduced checkpoint before raw scoring.
pub use joint::Result;
pub use pairs::MAX_WORK;
pub mod disk;
pub mod evaluation;
pub mod model;
pub mod resources;
pub mod runner;
pub mod tensor;
pub const RAM_LIMIT_BYTES: u64 = 110_000_000_000;
#[cfg(test)]
mod tests;

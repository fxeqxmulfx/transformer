//! CM56 at b5f1c9d7: complete complex class-coordinate heads with a learned
//! positive margin. Coordinates are an explicit identifying prior. Discovery
//! precedes the convex CE stage; no finite-precision correctness proof is claimed.
pub use convex_stress_frequency_data as frequency;
pub use convex_stress_prefix_programs as parent;
pub use parent::{C, Input, Observation, Result, argmax, ce, data, hash, metrics};
use serde::{Deserialize, Serialize};
pub mod coordinates;
pub mod discovery;
#[cfg(test)]
mod discovery_tests;
pub mod evaluation;
pub mod head;
pub mod model;
pub mod program;
pub mod runner;
#[cfg(test)]
mod tests;
pub mod training;
pub const PROTOCOL: &str = include_str!("../protocol.json");

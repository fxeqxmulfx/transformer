//! CM55 (Lean f0f2d385): fixed affine sharing with complete coefficient costs.
//! Geometry/decoder discovery is a declared training-only constructor, separate
//! from convex ordinary CE in its fixed, realified complex parameters.
pub use convex_stress_shared_head::training::{Mode, Policy};
pub use convex_stress_shared_head::{
    C, Input, Record, Result, ce_gradient, cross_entropy, digest, hash,
};
use serde::{Deserialize, Serialize};
pub mod features;
pub mod fit;
pub mod inspection;
pub mod limits;
pub mod model;
pub mod projection;
#[cfg(test)]
mod tests;
pub mod training;
pub use model::Model;
pub const PROTOCOL: &str = include_str!("../protocol.json");
pub const RADIUS: f64 = 1_048_576.;
pub const COMMON: f64 = 192.;
pub const RIDGES: [f64; 3] = [0.000001, 0.0001, 0.01];
pub const GAINS: [f64; 4] = [2., 8., 32., 128.];
pub const DIRECTIONS: [[f64; 2]; 16] = [
    [1., 0.],
    [0., 1.],
    [1., 1.],
    [1., -1.],
    [2., 1.],
    [1., 2.],
    [2., -1.],
    [1., -2.],
    [3., 1.],
    [1., 3.],
    [3., -1.],
    [1., -3.],
    [3., 2.],
    [2., 3.],
    [3., -2.],
    [2., -3.],
];
pub fn protocol_hash() -> Result<String> {
    digest(&serde_json::from_str::<serde_json::Value>(PROTOCOL)?)
}

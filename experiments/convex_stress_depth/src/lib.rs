//! CD3/CD5/CD8: fixed complex causal paths and one learned output tensor.
//! Lean JointHead.lean at 32a9b668 proves the coefficient domain and ordinary CE
//! convex. Numerical feature construction and optimization require native checks.
pub use convex_stress_hull::spectral::C;
pub use convex_stress_hull::{Result, data, hash, metrics};
pub mod checkpoint;
pub mod cohort;
pub mod features;
pub mod head;
pub mod independent;
pub mod training;

//! CM86--88 complete learned rectangular sequence model and native updates.
//!
//! Source: Lean LearnedBilinear and its Identification/Updates modules.
//! Exact zero storage is justified by Updates.Support; numerical SVD and f64
//! CE remain approximations checked by independent reconstruction controls.
pub use baseline::Result;
pub use num_complex::Complex64 as C;
pub mod algebra;
pub mod domain;
pub mod experiment;
pub mod guard;
pub mod model;
pub mod scaling;
pub mod train;

#[cfg(test)]
mod tests;

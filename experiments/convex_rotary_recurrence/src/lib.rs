//! CM97/98 complete learned two-state complex recurrence, with learned rotary positions.
//! Joint function mixtures are convex; numerical component discovery is approximate.
pub use bilinear::{C, Result};
pub mod atom;
mod coordinates;
pub mod experiment;
pub mod model;
pub mod proposal;
pub mod train;

#[cfg(test)]
mod tests;

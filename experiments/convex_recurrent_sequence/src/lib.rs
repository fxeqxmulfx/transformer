//! CM95/96 bounded learned scalar complex recurrence on original object sequences.
//! The convex function domain includes all components; discovery is approximate.
pub use bilinear::{C, Result};
pub mod experiment;
pub mod model;
pub mod proposal;
pub mod train;

#[cfg(test)]
mod tests;

//! CM106 at3c4705fb: learned-content revisits with complete historical caches.
//! Content search is approximate and uses finite actual CE evaluations.
pub use rotary::{C, Result};
pub mod cache;
pub mod content;
pub mod diagnostic;
pub mod experiment;
pub mod history;
pub mod score;
pub mod source;
#[cfg(test)]
mod tests;
pub mod train;
pub mod update;

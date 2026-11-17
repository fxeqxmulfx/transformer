//! CM100: revisit complete readouts on states learned by the original recurrence.
//! Conditional refitting is a controlled optimizer study, without unseen-label guarantees.
pub use rotary::{C, Result};
pub mod continuation;
pub mod experiment;
pub mod score;
pub mod source;
#[cfg(test)]
mod tests;
pub mod update;

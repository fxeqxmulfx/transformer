//! Fixed generic pair-state family for CM64. Only vocabulary and original
//! observed (input,input,raw-label) rows cross the model boundary. The family
//! is a structural prior, independent of the flat-spectrum capacity bound.
pub use joint::{Result, Work};
use serde::{Deserialize, Serialize};
pub mod family;
pub mod selection;
#[cfg(test)]
mod tests;

pub const FAMILY: &str = "pair-laurent3-two-terms-integer14-v1";
pub const PROGRAMS: usize = 4767;
pub const PROTOCOL: &str = include_str!("../protocol.json");

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Training {
    pub vocabulary: usize,
    pub rows: Vec<[u32; 3]>,
}
impl Training {
    pub fn validate(&self) -> Result<()> {
        family::Evaluator::new(self.vocabulary)?;
        if self.rows.is_empty()
            || self.rows.len() > 65536
            || self
                .rows
                .iter()
                .flatten()
                .any(|&v| v as usize >= self.vocabulary)
        {
            return Err("invalid original pair training records".into());
        }
        Ok(())
    }
}
pub fn digest(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    use std::fmt::Write;
    let mut text = String::with_capacity(64);
    for x in Sha256::digest(bytes) {
        write!(&mut text, "{x:02x}").unwrap();
    }
    text
}

pub fn policy(mode: joint::training::Mode) -> joint::training::Config {
    use joint::training::Mode::*;
    joint::training::Config {
        mode,
        epochs: match mode {
            Forward => 0,
            Full => 4,
            Mini => 1,
        },
        batch_size: 127,
        learning_rate: 1e6,
        initial_mass: 8.,
    }
}

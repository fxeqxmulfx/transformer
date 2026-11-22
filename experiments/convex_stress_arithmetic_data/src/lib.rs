//! Native data oracles for the54 original arithmetic cases. These functions
//! belong only to data preparation; they are never learned model features.
use sha2::{Digest, Sha256};
use std::fmt::Write;
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub fn hash(bytes: &[u8]) -> String {
    let mut text = String::with_capacity(64);
    for byte in Sha256::digest(bytes) {
        write!(&mut text, "{byte:02x}").unwrap();
    }
    text
}
pub mod dataset;
pub mod inventory;
pub mod oracle;
mod registry;
// Preserve the reviewed original22 arithmetic split, fingerprint and oracle
// implementation. Independent tests compare the newly ported oracle to it.
// Its historical RNG method returns a word, not an Iterator item.
#[allow(clippy::should_implement_trait)]
#[path = "../../convex_stress/src/vendor/arithmetic_data.rs"]
pub mod reference;
pub const CASES: &str = include_str!("../references/cases.json");
pub const DEFINITIONS: &str = include_str!("../references/definitions.json");
pub const ORIGINAL35: &str = include_str!("../../convex_stress/suite.json");
pub const PROTOCOL: &str = include_str!("../protocol.json");
#[cfg(test)]
mod tests;

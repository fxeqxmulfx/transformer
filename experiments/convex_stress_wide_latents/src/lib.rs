//! Complete fixed-family extension after CM67 f1783d04, with the CM65/66
//! reduced full joint domain. Family generation consumes no observations.
//! Flat spectrum bounds the complete complex head, not target realizability.
pub use joint::{Result, Work};
pub use pairs::{Training, digest, policy};
use serde::{Deserialize, Serialize};
pub mod encoding;
pub mod family;
pub mod selection;
#[cfg(test)]
mod tests;

pub const FAMILY: &str = "pair-laurent3-three-terms-integer14-v1";
pub const PROGRAMS: usize = 299_551;
pub const MAX_WORK: u64 = 40_000_000_000_000;
pub const PROTOCOL: &str = include_str!("../protocol.json");
pub const REPORT_LIMIT: u64 = 64 * 1024 * 1024;
pub const COHORT_RESERVE: u64 = 2 * 48 * (REPORT_LIMIT + 8 * 1024 * 1024) + 1024u64.pow(3);

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut result = String::with_capacity(2 * bytes.len());
    for b in bytes {
        write!(&mut result, "{b:02x}").unwrap();
    }
    result
}

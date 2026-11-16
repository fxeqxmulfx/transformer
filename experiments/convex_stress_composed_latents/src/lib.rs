//! Fixed input compositions under CM67's full-domain inclusion theorem at
//! f1783d04, rechecked before this native change. CM65/66/68 retain the full
//! convex decoder, weighted projection and original packed observations.
//! The structural input prior is distinct from the flat-spectrum head bound.
pub use joint::{Result, Work};
pub use pairs::{Training, digest, policy};
use serde::{Deserialize, Serialize};
pub mod family;
// Reuse the reviewed complete partition and materialization engine from
// e5ef378a. Its crate-local family resolves to this strictly larger bank.
#[path = "../../convex_stress_wide_latents/src/encoding.rs"]
pub mod encoding;
#[path = "../../convex_stress_wide_latents/src/selection.rs"]
pub mod selection;
#[cfg(test)]
mod tests;

pub const FAMILY: &str = "fixed-branch-residue-digit-four-terms-v1";
pub const PROGRAMS: usize = 962_806;
pub const MAX_WORK: u64 = wide::MAX_WORK;
pub const PROTOCOL: &str = include_str!("../protocol.json");
pub const REPORT_LIMIT: u64 = 128 * 1024 * 1024;
pub const COHORT_RESERVE: u64 = 2 * 48 * (REPORT_LIMIT + 8 * 1024 * 1024) + 1024u64.pow(3);

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut result = String::with_capacity(2 * bytes.len());
    for b in bytes {
        write!(&mut result, "{b:02x}").unwrap();
    }
    result
}

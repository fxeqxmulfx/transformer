//! Compare actual CM34 complete-kernel heads with complete prior evidence.
//! Ports refined_compare ata0046702. No new training/discovery; both parent
//! arms and original primary-success semantics remain unchanged.
use convex_stress_kernel::{self as component, Result, binding, catalogue, discovery, training};
use convex_stress_quotient_cohort::artifacts::{digest, hash, read, write};
use convex_stress_relabel::{data, metrics};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};
pub mod comparison;
mod execute;
mod imported;
mod inputs;
mod outcomes;
mod parents;
mod summary;
#[cfg(test)]
mod tests;
pub use execute::{prepare, replay_case, run_case};
pub use parents::{Reference, Review};
pub use summary::summarize;
mod sources {
    include!(concat!(env!("OUT_DIR"), "/sources.rs"));
}
const PROTOCOL: &str = include_str!("../protocol.json");
const SUITE: &str = include_str!("../../convex_stress/suite.json");
pub(crate) const MODES: [&str; 3] = ["full_batch", "minibatch", "forward_only"];
pub(crate) const POLICIES: [&str; 2] = ["minimum_ce", "first_small_ce"];
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub producer_commit: String,
    pub suite_sha256: String,
    pub parent: Reference,
    pub staging: Reference,
}
fn protocol() -> Result<Value> {
    Ok(serde_json::from_str(PROTOCOL)?)
}
fn suite() -> Result<Value> {
    Ok(serde_json::from_str(SUITE)?)
}
fn commit(v: &str) -> bool {
    v.len() == 40 && v.bytes().all(|b| b.is_ascii_hexdigit())
}
fn sha(v: &str) -> bool {
    v.len() == 64 && v.bytes().all(|b| b.is_ascii_hexdigit())
}
fn name(v: &str) -> Result<&str> {
    if v.is_empty()
        || !v
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
    {
        return Err("invalid local task name".into());
    }
    Ok(v)
}
fn tasks(suite: &Value) -> Result<Vec<String>> {
    let all = suite["tasks"]
        .as_array()
        .ok_or("missing original suite")?
        .iter()
        .map(|t| Ok(name(t["name"].as_str().ok_or("missing task")?)?.to_string()))
        .collect::<Result<Vec<_>>>()?;
    if all.len() != 35 || all.iter().collect::<BTreeSet<_>>().len() != 35 {
        return Err("complete distinct35-task scope required".into());
    }
    Ok(all)
}
fn mode(m: training::Mode) -> &'static str {
    match m {
        training::Mode::Full => MODES[0],
        training::Mode::Mini => MODES[1],
        training::Mode::Forward => MODES[2],
    }
}
fn passes(metrics: &Value, primary: &str) -> Result<(bool, bool)> {
    let splits = metrics.as_object().ok_or("missing metrics")?;
    let accuracy = |v: &Value| {
        v["sequence_accuracy"]
            .as_f64()
            .filter(|v| v.is_finite() && (0. ..=1.).contains(v))
            .ok_or("invalid sequence accuracy")
    };
    let main = accuracy(&metrics[primary])?;
    let heldout = splits
        .iter()
        .filter(|(s, _)| s.as_str() != "train")
        .map(|(_, m)| accuracy(m))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if heldout.is_empty() {
        return Err("missing held-out metric scope".into());
    }
    Ok((main >= 0.99, heldout.iter().all(|&p| p >= 0.99)))
}
pub fn fingerprint() -> String {
    let mut bytes = vec![];
    for &(name, source) in sources::SOURCES {
        bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
        bytes.extend_from_slice(name.as_bytes());
        bytes.extend_from_slice(&(source.len() as u64).to_le_bytes());
        bytes.extend_from_slice(source);
    }
    bytes.extend_from_slice(convex_stress_kernel_cohort::fingerprint().as_bytes());
    bytes.extend_from_slice(convex_stress_refined_compare::fingerprint().as_bytes());
    hash(&bytes)
}

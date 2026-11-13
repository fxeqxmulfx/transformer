//! CFt4/CD8 causal proposals with a complete, reviewed quotient comparison.
use artifacts::{digest, hash, read, write};
use convex_stress_causal_pairs::{self as component, Result, binding, training};
pub use convex_stress_quotient_cohort::artifacts;
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
mod inputs;
mod inventory;
mod outcomes;
mod parent;
mod summary;
#[cfg(test)]
mod tests;
pub use execute::{replay_case, run_case};
pub use inventory::prepare;
pub use parent::{Parent, Review};
pub use summary::summarize;
mod sources {
    include!(concat!(env!("OUT_DIR"), "/sources.rs"));
}
const PROTOCOL: &str = include_str!("../protocol.json");
const SUITE: &str = include_str!("../../../convex_stress/suite.json");
const MODES: [&str; 3] = ["full_batch", "minibatch", "forward_only"];
const POLICIES: [&str; 2] = ["minimum_ce", "first_small_ce"];
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub producer_commit: String,
    pub suite_sha256: String,
    pub parent: Parent,
}
pub fn fingerprint() -> String {
    let mut bytes = vec![];
    for &(name, source) in sources::SOURCES {
        bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
        bytes.extend_from_slice(name.as_bytes());
        bytes.extend_from_slice(&(source.len() as u64).to_le_bytes());
        bytes.extend_from_slice(source);
    }
    let inherited = convex_stress_quotient_cohort::fingerprint();
    bytes.extend_from_slice(inherited.as_bytes());
    hash(&bytes)
}
fn commit(s: &str) -> bool {
    s.len() == 40 && s.bytes().all(|b| b.is_ascii_hexdigit())
}
fn protocol() -> Result<Value> {
    Ok(serde_json::from_str(PROTOCOL)?)
}
fn suite() -> Result<Value> {
    Ok(serde_json::from_str(SUITE)?)
}
fn name(s: &str) -> Result<&str> {
    if s.is_empty()
        || !s
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
    {
        return Err("task must have a plain local name".into());
    }
    Ok(s)
}
fn tasks(suite: &Value) -> Result<Vec<String>> {
    let all = suite["tasks"]
        .as_array()
        .ok_or("missing original suite")?
        .iter()
        .map(|r| Ok(name(r["name"].as_str().ok_or("missing original task name")?)?.to_string()))
        .collect::<Result<Vec<_>>>()?;
    if all.len() != 35 || all.iter().collect::<BTreeSet<_>>().len() != 35 {
        return Err("exactly35 distinct original tasks required".into());
    }
    Ok(all)
}
fn mode(mode: training::Mode) -> &'static str {
    match mode {
        training::Mode::Full => MODES[0],
        training::Mode::Mini => MODES[1],
        training::Mode::Forward => MODES[2],
    }
}
fn passes(metrics: &Value, primary: &str) -> Result<(bool, bool)> {
    let fields = metrics
        .as_object()
        .ok_or("missing complete original metrics")?;
    let mut heldout = vec![];
    for (split, v) in fields {
        if split != "train" {
            heldout.push(
                v["sequence_accuracy"]
                    .as_f64()
                    .filter(|x| x.is_finite() && (0. ..=1.).contains(x))
                    .ok_or("invalid held-out accuracy")?,
            );
        }
    }
    let primary = metrics[primary]["sequence_accuracy"]
        .as_f64()
        .filter(|x| x.is_finite() && (0. ..=1.).contains(x))
        .ok_or("invalid primary accuracy")?;
    if heldout.is_empty() {
        return Err("missing original held-out splits".into());
    }
    Ok((primary >= 0.99, heldout.iter().all(|x| *x >= 0.99)))
}

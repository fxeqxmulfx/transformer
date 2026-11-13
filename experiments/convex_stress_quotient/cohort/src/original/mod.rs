//! Original full35 extension, with independently reviewed parents and no answer features.
use crate::*;
use convex_stress_relabel::{data, metrics};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
};
mod candidates;
mod execute;
#[cfg(test)]
mod fixtures;
mod inputs;
mod inventory;
mod parent;
mod summary;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod validation_tests;
pub use execute::{replay_case, run_case};
pub use inventory::prepare;
pub use parent::{Parent, Review};
pub use summary::summarize;
const ORIGINAL_PROTOCOL: &str = include_str!("../../original_protocol.json");
const SUITE: &str = include_str!("../../../../convex_stress/suite.json");
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub producer_commit: String,
    pub suite_sha256: String,
    pub parent: Parent,
}
fn protocol() -> Result<Value> {
    Ok(serde_json::from_str(ORIGINAL_PROTOCOL)?)
}
fn planned() -> Result<Vec<String>> {
    Ok(serde_json::from_value(protocol()?["tasks"].clone())?)
}
fn suite() -> Result<Value> {
    Ok(serde_json::from_str(SUITE)?)
}
fn tasks(suite: &Value) -> Result<Vec<String>> {
    suite["tasks"]
        .as_array()
        .ok_or("missing original suite")?
        .iter()
        .map(|v| {
            v["name"]
                .as_str()
                .map(String::from)
                .ok_or_else(|| "missing task name".into())
        })
        .collect()
}
fn name(value: &str) -> Result<&str> {
    if value.is_empty()
        || !value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
    {
        return Err("task must have a plain local name".into());
    }
    Ok(value)
}
fn passes(metrics: &Value, primary: &str) -> Result<(bool, bool)> {
    let fields = metrics
        .as_object()
        .ok_or("missing actual original metrics")?;
    let heldout: Vec<_> = fields
        .iter()
        .filter(|(s, _)| s.as_str() != "train")
        .map(|(_, v)| {
            v["sequence_accuracy"]
                .as_f64()
                .filter(|x| x.is_finite() && *x >= 0. && *x <= 1.)
                .ok_or("invalid held-out accuracy")
        })
        .collect::<std::result::Result<_, _>>()?;
    let p = metrics[primary]["sequence_accuracy"]
        .as_f64()
        .filter(|x| x.is_finite() && *x >= 0. && *x <= 1.)
        .ok_or("missing primary split")?;
    if heldout.is_empty() {
        return Err("missing original held-out splits".into());
    }
    Ok((p >= 0.99, heldout.iter().all(|x| *x >= 0.99)))
}

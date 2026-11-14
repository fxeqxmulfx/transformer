//! The reviewed 276 comparison with CM41 saved-parent availability routing.
//! Source imports retain the unchanged selection, state lifecycle and reporting.
//! Only the causal wrapper and actual route diagnostics change prospectively.
use component::{binding, catalogue, discovery, training};
use convex_stress_kernel_compare::comparison as selection;
pub use convex_stress_kernel_compare::{Reference, Review};
use convex_stress_quotient_cohort::artifacts::{digest, hash, read, write};
use convex_stress_relabel::{data, metrics};
use convex_stress_shared_role_extension::{self as extension, Result, kernel as component, shared};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};
mod causal;
#[path = "../../convex_stress_shared_compare/src/diagnostics.rs"]
mod diagnostics;
#[path = "../../convex_stress_shared_compare/src/execute.rs"]
mod execute;
#[path = "../../convex_stress_shared_compare/src/inputs.rs"]
mod inputs;
#[path = "../../convex_stress_shared_compare/src/outcomes.rs"]
mod outcomes;
#[path = "../../convex_stress_shared_compare/src/parents.rs"]
mod parents;
#[path = "../../convex_stress_shared_compare/src/raw/mod.rs"]
mod raw;
#[path = "../../convex_stress_shared_compare/src/selected/mod.rs"]
mod selected;
#[path = "../../convex_stress_shared_compare/src/summary.rs"]
mod summary;
#[cfg(test)]
mod tests;
pub use execute::{prepare, replay_case, run_case};
pub use summary::summarize;
mod sources {
    include!(concat!(env!("OUT_DIR"), "/sources.rs"));
}
const PROTOCOL: &str = include_str!("../protocol.json");
const SUITE: &str = include_str!("../../convex_stress/suite.json");
const MODES: [&str; 3] = ["full_batch", "minibatch", "forward_only"];
const POLICIES: [&str; 2] = ["minimum_ce", "first_small_ce"];
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub producer_commit: String,
    pub suite_sha256: String,
    pub parents: BTreeMap<String, Reference>,
    pub checkpoint_root: PathBuf,
}
fn protocol() -> Result<Value> {
    let value: Value = serde_json::from_str(PROTOCOL)?;
    if value["reviewed_orchestration"]["source_fingerprint"]
        != convex_stress_shared_compare::fingerprint()
        || value["partial_parent_component"]["routing_policy"]
            != convex_stress_partial_fallback::ROUTING_POLICY
    {
        return Err("reviewed orchestration or fixed partial-parent route changed".into());
    }
    Ok(value)
}
fn suite() -> Result<Value> {
    Ok(serde_json::from_str(SUITE)?)
}
fn planned() -> Result<Vec<String>> {
    Ok(serde_json::from_value(protocol()?["planned"].clone())?)
}
fn sha(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}
fn commit(s: &str) -> bool {
    s.len() == 40 && s.bytes().all(|b| b.is_ascii_hexdigit())
}
fn name(s: &str) -> Result<&str> {
    if s.is_empty()
        || !s
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
    {
        return Err("task must be a plain local name".into());
    }
    Ok(s)
}
fn tasks(suite: &Value) -> Result<Vec<String>> {
    let names = suite["tasks"]
        .as_array()
        .ok_or("missing original suite")?
        .iter()
        .map(|r| {
            r["name"]
                .as_str()
                .map(String::from)
                .ok_or_else(|| "missing original task name".into())
        })
        .collect::<Result<Vec<_>>>()?;
    if names.len() != 35 || names.iter().collect::<BTreeSet<_>>().len() != 35 {
        return Err("exactly 35 distinct original tasks required".into());
    }
    for t in &names {
        name(t)?;
    }
    Ok(names)
}
fn mode(m: training::Mode) -> &'static str {
    match m {
        training::Mode::Full => MODES[0],
        training::Mode::Mini => MODES[1],
        training::Mode::Forward => MODES[2],
    }
}
pub fn fingerprint() -> String {
    let mut bytes = vec![];
    for &(name, source) in sources::SOURCES {
        bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
        bytes.extend_from_slice(name.as_bytes());
        bytes.extend_from_slice(&(source.len() as u64).to_le_bytes());
        bytes.extend_from_slice(source);
    }
    for f in [
        convex_stress_shared_compare::fingerprint(),
        convex_stress_kernel_compare::fingerprint(),
        convex_stress_shared_roles_cohort::fingerprint(),
        convex_stress_causal_pairs_cohort::fingerprint(),
    ] {
        bytes.extend_from_slice(f.as_bytes());
    }
    hash(&bytes)
}
fn cell<'a>(parent: &'a Value, policy: &str, task: &str, mode: &str) -> Result<&'a Value> {
    let arms = parent["arms"]
        .as_array()
        .ok_or("missing parent arms")?
        .iter()
        .filter(|r| r["initializer"] == policy)
        .collect::<Vec<_>>();
    if arms.len() != 1 {
        return Err("initializer missing or repeated".into());
    }
    let rows = arms[0]["choices"]
        .as_array()
        .ok_or("missing parent choices")?
        .iter()
        .filter(|r| r["task"] == task && r["training_mode"] == mode)
        .collect::<Vec<_>>();
    if rows.len() != 1 {
        return Err("task/mode missing or repeated".into());
    }
    Ok(rows[0])
}
fn passes(metrics: &Value, primary: &str) -> Result<(bool, bool)> {
    let scores = metrics.as_object().ok_or("missing original metrics")?;
    if !scores.contains_key(primary) {
        return Err("primary split omitted".into());
    }
    let accuracy = |m: &Value| -> Result<f64> {
        m["sequence_accuracy"]
            .as_f64()
            .filter(|x| x.is_finite() && (0. ..=1.).contains(x))
            .ok_or_else(|| "invalid original sequence accuracy".into())
    };
    let heldout = scores
        .iter()
        .filter(|(s, _)| s.as_str() != "train")
        .map(|(_, m)| accuracy(m))
        .collect::<Result<Vec<_>>>()?;
    if heldout.is_empty() {
        return Err("held-out metric scope omitted".into());
    }
    Ok((
        accuracy(&scores[primary])? >= 0.99,
        heldout.iter().all(|&x| x >= 0.99),
    ))
}

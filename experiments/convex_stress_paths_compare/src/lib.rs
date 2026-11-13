//! Reviewed fixed-feature endpoints; selection has no access to held-out metrics.
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
mod artifacts;
mod consequences;
mod depth;
mod import;
mod report;
mod selection;
mod transformer;
mod sources {
    include!(concat!(env!("OUT_DIR"), "/sources.rs"));
}
#[cfg(test)]
mod fixtures;
#[cfg(test)]
mod tests;
use artifacts::{Artifact, Tracker, digest, hash, read, write};
pub use import::{Request, prepare, replay};
const SUITE: &str = include_str!("../../convex_stress/suite.json");
const PROTOCOL: &str = include_str!("../protocol.json");
const MODES: [&str; 3] = ["full_batch", "minibatch", "forward_only"];
const POLICIES: [&str; 2] = ["minimum_ce", "first_small_ce"];
const TRANSPORTS: [&str; 2] = ["Residual", "Fourier"];
fn text<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    v[key]
        .as_str()
        .ok_or_else(|| format!("missing string {key}").into())
}
fn rows<'a>(v: &'a Value, key: &str) -> Result<&'a Vec<Value>> {
    v[key]
        .as_array()
        .ok_or_else(|| format!("missing rows {key}").into())
}
fn number(v: &Value, key: &str) -> Result<f64> {
    v[key]
        .as_f64()
        .filter(|x| x.is_finite() && *x >= 0.)
        .ok_or_else(|| format!("invalid nonnegative {key}").into())
}
fn check(ok: bool, why: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(why.into()) }
}
fn protocol() -> Result<Value> {
    Ok(serde_json::from_str(PROTOCOL)?)
}
pub fn fingerprint() -> String {
    let mut bytes = vec![];
    for &(name, source) in sources::SOURCES {
        bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
        bytes.extend_from_slice(name.as_bytes());
        bytes.extend_from_slice(&(source.len() as u64).to_le_bytes());
        bytes.extend_from_slice(source);
    }
    hash(&bytes)
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Offer {
    reference: String,
    task: String,
    mode: String,
    policy: Option<String>,
    transport: Option<String>,
    train_ce: Option<f64>,
    primary: String,
    metrics: Value,
    saved: Value,
}
#[derive(Default)]
struct Loaded {
    prior: Vec<Offer>,
    retained: Vec<Offer>,
    transformers: Vec<Value>,
    depth_records: Vec<Value>,
    evidence: Tracker,
    snapshots: BTreeMap<String, Artifact>,
    accounting: BTreeMap<String, Value>,
}
fn task<'a>(suite: &'a Value, name: &str) -> Result<&'a Value> {
    let matching: Vec<_> = rows(suite, "tasks")?
        .iter()
        .filter(|r| r["name"] == name)
        .collect();
    check(matching.len() == 1, "unknown or duplicated original task")?;
    Ok(matching[0])
}
fn data(suite: &Value, name: &str, fingerprints: &Value) -> Result<()> {
    check(
        !fingerprints.is_null() && task(suite, name)?["data_fingerprints"] == *fingerprints,
        "original data fingerprints changed",
    )
}

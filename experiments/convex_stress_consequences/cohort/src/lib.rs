//! Frozen original comparison for CM29 bounded categorical code discovery.
use convex_stress_consequences::{Result, learner, view};
use convex_stress_relabel::{data, hash, metrics};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{fs, path::Path};
mod sources {
    include!(concat!(env!("OUT_DIR"), "/sources.rs"));
}
mod artifacts;
mod evaluate;
#[cfg(test)]
mod fixtures;
mod inputs;
mod inventory;
mod parent;
mod run;
mod selection;
mod summary;
#[cfg(test)]
mod tests;
use artifacts::{digest, read, write};
use inputs::{Inputs, check_input, inputs};
pub use inventory::prepare;
pub use run::{replay, run};
pub use summary::summarize;
const PROTOCOL: &str = include_str!("../protocol.json");
const SUITE: &str = include_str!("../../../convex_stress/suite.json");
const LIMIT: u64 = 8 * 1024 * 1024 * 1024;
fn protocol() -> Result<Value> {
    Ok(serde_json::from_str(PROTOCOL)?)
}
fn tasks() -> Result<Vec<String>> {
    Ok(serde_json::from_value(protocol()?["tasks"].clone())?)
}
fn modes() -> Result<Vec<String>> {
    Ok(serde_json::from_value(
        protocol()?["reference_modes"].clone(),
    )?)
}
fn policy() -> Result<learner::Policy> {
    Ok(serde_json::from_value(protocol()?["learner"].clone())?)
}
pub fn fingerprint() -> String {
    let mut bytes = vec![];
    for &(name, source) in sources::SOURCES {
        bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
        bytes.extend_from_slice(name.as_bytes());
        bytes.extend_from_slice(&(source.len() as u64).to_le_bytes());
        bytes.extend_from_slice(source);
    }
    bytes.extend_from_slice(convex_stress_decode::cohort::fingerprint().as_bytes());
    bytes.extend_from_slice(convex_stress_relabel::cohort::fingerprint().as_bytes());
    hash(&bytes)
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Candidate {
    view: usize,
    candidate: usize,
    duplicate_of: Option<usize>,
    model_sha256: String,
    output_sha256: String,
    value: view::Candidate,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct ViewRecord {
    index: usize,
    artifact: artifacts::Artifact,
    disposition: view::Disposition,
    candidates: usize,
    selected: Option<usize>,
    work: view::Work,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Bank {
    views: Vec<ViewRecord>,
    candidates: Vec<Candidate>,
}

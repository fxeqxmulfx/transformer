//! Training-only refinement of immutable, already replayed categorical discoveries.
use convex_stress_quotient_cohort as parent;
use convex_stress_refined_quotient::{Result, binding, quotient, training};
use parent::artifacts::{digest, hash, read, write};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
};
mod inputs;
mod source;
mod stage;
mod summary;
#[cfg(test)]
mod tests;
pub use source::Reference;
pub use stage::{replay_case, run_case};
pub use summary::summarize;
mod sources {
    include!(concat!(env!("OUT_DIR"), "/sources.rs"));
}
const SUITE: &str = include_str!("../../../convex_stress/suite.json");
const PROTOCOL: &str = include_str!("../protocol.json");
fn protocol() -> Result<Value> {
    Ok(serde_json::from_str(PROTOCOL)?)
}
fn suite() -> Result<Value> {
    Ok(serde_json::from_str(SUITE)?)
}
fn planned() -> Result<Vec<String>> {
    Ok(serde_json::from_value(protocol()?["tasks"].clone())?)
}
fn tasks(suite: &Value) -> Result<Vec<String>> {
    suite["tasks"]
        .as_array()
        .ok_or("missing suite")?
        .iter()
        .map(|r| {
            r["name"]
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
        return Err("task must be a plain local name".into());
    }
    Ok(value)
}
fn commit(s: &str) -> bool {
    s.len() == 40 && s.bytes().all(|b| b.is_ascii_hexdigit())
}
pub fn fingerprint() -> String {
    let mut bytes = vec![];
    for &(name, source) in sources::SOURCES {
        bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
        bytes.extend_from_slice(name.as_bytes());
        bytes.extend_from_slice(&(source.len() as u64).to_le_bytes());
        bytes.extend_from_slice(source);
    }
    bytes.extend_from_slice(parent::fingerprint().as_bytes());
    hash(&bytes)
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub producer_commit: String,
    pub suite_sha256: String,
    pub discovery: Reference,
}
fn manifest(
    request: &Request,
    suite: &Value,
    planned: &[String],
    records: &[Value],
    sources: &[Value],
) -> Result<Value> {
    if !commit(&request.producer_commit)
        || request.suite_sha256 != hash(SUITE.as_bytes())
        || planned.is_empty()
        || records.len() != planned.len()
        || sources.len() != planned.len()
        || planned
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != planned.len()
    {
        return Err("invalid refined stage producer, data or complete scope".into());
    }
    let all = tasks(suite)?;
    for (i, task) in planned.iter().enumerate() {
        name(task)?;
        if !all.contains(task) || records[i]["task"] != *task || sources[i]["task"] != *task {
            return Err("refined input or source task order changed".into());
        }
    }
    Ok(
        json!({"phase":"training_only", "source_fingerprint":fingerprint(),
        "binary_sha256":digest(&std::env::current_exe()?)?, "request":request,
        "protocol":protocol()?, "tasks":all, "planned":planned,
        "inputs":records,"discoveries":sources,"heldout_evaluation":false,"parent_comparison":false}),
    )
}
pub fn prepare(root: &Path, request: Request) -> Result<()> {
    if request.suite_sha256 != hash(SUITE.as_bytes()) {
        return Err("original suite pin differs".into());
    }
    let suite = suite()?;
    let planned = planned()?;
    let sources = source::prepare(&request.discovery, &planned, &tasks(&suite)?)?;
    let inputs = planned
        .iter()
        .map(|t| Ok(inputs::build(&suite, t)?.record))
        .collect::<Result<Vec<_>>>()?;
    prepare_with(root, request, &suite, &planned, inputs, sources)
}
fn prepare_with(
    root: &Path,
    request: Request,
    suite: &Value,
    planned: &[String],
    records: Vec<Value>,
    sources: Vec<Value>,
) -> Result<()> {
    if root.exists() {
        return Err("fresh refined staging directory required".into());
    }
    let inv = manifest(&request, suite, planned, &records, &sources)?;
    fs::create_dir(root)?;
    write(&root.join("request.json"), &request)?;
    write(&root.join("inventory.json"), &inv)
}
fn inventory_with(root: &Path, suite: &Value, planned: &[String]) -> Result<Value> {
    let request: Request = read(&root.join("request.json"))?;
    let inv: Value = read(&root.join("inventory.json"))?;
    if inv
        != manifest(
            &request,
            suite,
            planned,
            inv["inputs"].as_array().ok_or("missing refined inputs")?,
            inv["discoveries"]
                .as_array()
                .ok_or("missing source records")?,
        )?
    {
        return Err("refined inventory, source, executable or protocol changed".into());
    }
    Ok(inv)
}
fn inventory(root: &Path) -> Result<Value> {
    let inv = inventory_with(root, &suite()?, &planned()?)?;
    source::pins(&serde_json::from_value(
        inv["request"]["discovery"].clone(),
    )?)?;
    Ok(inv)
}

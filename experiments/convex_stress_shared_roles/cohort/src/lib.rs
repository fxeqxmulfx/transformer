//! CM40 training-only adapter, ported from readout_kernel/cohort at 5d4eee84.
//! Original output partitions and categorical role identities are construction
//! metadata. Separate original records preserve multiplicity and order; only
//! identical algebraic equations are coalesced. No held-out score is computed.
use convex_stress_quotient_cohort as parent;
use convex_stress_shared_roles::{Result, binding, discovery, kernel, records, training};
use kernel::catalogue;
use parent::artifacts::{digest, hash, read, write};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{fs, path::Path};
mod evidence;
mod inputs;
mod stage;
mod summary;
#[cfg(test)]
mod tests;
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
        .ok_or("missing suite tasks")?
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
}
fn manifest(
    request: &Request,
    suite: &Value,
    planned: &[String],
    records: &[Value],
) -> Result<Value> {
    let all = tasks(suite)?;
    if !commit(&request.producer_commit)
        || request.suite_sha256 != hash(SUITE.as_bytes())
        || all.len() != 35
        || planned.is_empty()
        || records.len() != planned.len()
        || all.iter().collect::<std::collections::BTreeSet<_>>().len() != all.len()
        || planned
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != planned.len()
    {
        return Err("invalid kernel producer, data or complete task scope".into());
    }
    for (i, task) in planned.iter().enumerate() {
        name(task)?;
        if !all.contains(task) || records[i]["task"] != *task {
            return Err("kernel input task order changed".into());
        }
    }
    Ok(
        json!({"phase":"training_only","source_fingerprint":fingerprint(),
        "binary_sha256":digest(&std::env::current_exe()?)?,"request":request,
        "protocol":protocol()?,"tasks":all,"planned":planned,"primes":catalogue(),
        "inputs":records,"heldout_evaluation":false,"parent_comparison":false,
        "raw_label_identification":false,"target_readout_compatibility_proved":false,
        "shared_role_target_proved":false,"shared_role_constraints_used":true,
        "original_record_multiplicity_retained":true,
        "declared_readout_used_in_discovery":true,"original_raw_ce_unchanged":true}),
    )
}
pub fn prepare(root: &Path, request: Request) -> Result<()> {
    if request.suite_sha256 != hash(SUITE.as_bytes()) {
        return Err("original suite bytes differ".into());
    }
    let suite = suite()?;
    let planned = planned()?;
    let records = planned
        .iter()
        .map(|t| Ok(inputs::build(&suite, t)?.record))
        .collect::<Result<Vec<_>>>()?;
    prepare_with(root, request, &suite, &planned, records)
}
fn prepare_with(
    root: &Path,
    request: Request,
    suite: &Value,
    planned: &[String],
    records: Vec<Value>,
) -> Result<()> {
    if root.exists() {
        return Err("fresh kernel cohort directory required".into());
    }
    let inv = manifest(&request, suite, planned, &records)?;
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
            inv["inputs"].as_array().ok_or("missing original inputs")?,
        )?
    {
        return Err("kernel inventory, source, executable or protocol changed".into());
    }
    Ok(inv)
}
fn inventory(root: &Path) -> Result<Value> {
    inventory_with(root, &suite()?, &planned()?)
}

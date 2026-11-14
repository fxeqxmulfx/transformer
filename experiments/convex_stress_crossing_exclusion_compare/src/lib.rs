//! Original CM50 comparison, ported from reviewed task310 atdf79e005.
//! Keep all CM49/CM48 outcomes, guarded CE results and earlier CE tradeoffs.
use convex_stress_crossing_exclusion::{self as head, Result, records};
use convex_stress_relabel::{data, metrics};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{BufReader, Read},
    path::{Path, PathBuf},
};
mod attempt;
#[cfg(test)]
mod attempt_tests;
mod counts;
#[cfg(test)]
mod counts_tests;
mod import;
mod model;
#[cfg(test)]
mod model_tests;
mod parent;
mod reference;
#[cfg(test)]
mod reference_tests;
mod run;
mod summary;
#[cfg(test)]
mod tests;
pub use parent::prepare;
pub use run::{replay_case, run_case};
pub use summary::summarize;
mod sources {
    include!(concat!(env!("OUT_DIR"), "/sources.rs"));
}
const PROTOCOL: &str = include_str!("../protocol.json");
const SUITE: &str = include_str!("../../convex_stress/suite.json");
const CASE_WORK_CAP: u64 = 4_000_000_000_000;
const CASE_BUFFER_LIMIT: u64 = 56 * 1024 * 1024 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub producer_commit: String,
    pub parent_root: PathBuf,
    pub completed_manifest: PathBuf,
    pub parent_review_exit: PathBuf,
    pub parent_review_commit: PathBuf,
    pub component_review_exit: PathBuf,
    pub component_review_commit: PathBuf,
    pub local_reference_root: PathBuf,
    pub local_reference_manifest: PathBuf,
    pub local_reference_review_exit: PathBuf,
    pub local_reference_review_commit: PathBuf,
    pub previous_attempt_root: PathBuf,
    pub previous_attempt_manifest: PathBuf,
}

pub fn fingerprint() -> String {
    let mut h = Sha256::new();
    for (name, bytes) in sources::SOURCES {
        h.update(name.as_bytes());
        h.update(bytes);
    }
    h.update(convex_stress_raw_exclusion_compare::fingerprint());
    h.update(SUITE);
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}
fn digest(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        h.update(&buffer[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}
fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn read<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    Ok(serde_json::from_reader(BufReader::new(fs::File::open(
        path,
    )?))?)
}
fn write<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if path.exists() {
        return Err("refusing to overwrite an evidence artifact".into());
    }
    fs::write(path, serde_json::to_vec(value)?)?;
    Ok(())
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
fn protocol() -> Result<Value> {
    let p: Value = serde_json::from_str(PROTOCOL)?;
    if p["parent_source_fingerprint"] != convex_stress_raw_exclusion_compare::fingerprint() {
        return Err("compiled original parent dependencies changed".into());
    }
    Ok(p)
}
fn passes(scores: &Value, primary: &str) -> Result<(bool, bool)> {
    let all = scores.as_object().ok_or("missing original metrics")?;
    let accuracy = |v: &Value| {
        v["sequence_accuracy"]
            .as_f64()
            .filter(|x| x.is_finite() && (0. ..=1.).contains(x))
            .ok_or("invalid original accuracy")
    };
    let held = all
        .iter()
        .filter(|(s, _)| s.as_str() != "train")
        .map(|(_, m)| accuracy(m))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if held.is_empty() {
        return Err("missing original held-out scope".into());
    }
    Ok((
        accuracy(all.get(primary).ok_or("missing primary split")?)? >= 0.99,
        held.iter().all(|&x| x >= 0.99),
    ))
}

fn raw_accuracy(metric: &Value) -> Value {
    metric
        .get("native_token_accuracy")
        .or_else(|| metric.get("token_accuracy"))
        .cloned()
        .unwrap_or(Value::Null)
}

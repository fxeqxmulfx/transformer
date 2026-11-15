//! Task332, CM38/CM55 audit4e8f8171: fixed coverage around reviewed331 geometry.
//! All prior references and all new failures/regressions remain in the report.
use convex_stress_relabel::{data, metrics};
use convex_stress_shared_geometry::{self as fresh, Result};
use convex_stress_signed_kernel as old;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{BufReader, Read, Write},
    path::{Path, PathBuf},
};
mod audit;
mod costs;
#[path = "../../convex_stress_signed_compare/src/counts.rs"]
mod counts;
#[cfg(test)]
mod fixture;
#[path = "../../convex_stress_signed_compare/src/goals.rs"]
mod goals;
mod inputs;
mod model;
mod run;
mod summary;
#[cfg(test)]
mod tests;
mod sources {
    include!(concat!(env!("OUT_DIR"), "/sources.rs"));
}
pub use inputs::prepare;
pub use run::{replay_case, run_case};
pub use summary::summarize;
const PROTOCOL: &str = include_str!("../protocol.json");
const SUITE: &str = include_str!("../../convex_stress/suite.json");
const BUFFER_LIMIT: u64 = 56 * 1024 * 1024 * 1024;
const WORK_LIMIT: u64 = 10_000_000_000_000;
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub producer_commit: String,
    pub parent_root: PathBuf,
    pub parent_manifest: PathBuf,
    pub component_root: PathBuf,
    pub diff_gpt_history: PathBuf,
    pub comparison_root: PathBuf,
    pub comparison_manifest: PathBuf,
}
pub fn fingerprint() -> String {
    static FINGERPRINT: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    FINGERPRINT.get_or_init(compute_fingerprint).clone()
}
fn compute_fingerprint() -> String {
    let mut h = Sha256::new();
    for (name, bytes) in sources::SOURCES {
        h.update(name.as_bytes());
        h.update(bytes);
    }
    h.update(convex_stress_signed_compare::fingerprint());
    h.update(SUITE);
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}
fn protocol() -> Result<Value> {
    static PROTOCOL_VALUE: std::sync::OnceLock<std::result::Result<Value, String>> =
        std::sync::OnceLock::new();
    PROTOCOL_VALUE
        .get_or_init(|| check_protocol().map_err(|e| e.to_string()))
        .clone()
        .map_err(Into::into)
}
fn check_protocol() -> Result<Value> {
    let p: Value = serde_json::from_str(PROTOCOL)?;
    if p["parent_source_fingerprint"] != convex_stress_signed_compare::fingerprint() {
        return Err("reviewed parent dependency fingerprint changed".into());
    }
    Ok(p)
}
fn digest(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        h.update(&buffer[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}
fn read<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    Ok(serde_json::from_reader(BufReader::new(fs::File::open(
        path,
    )?))?)
}
fn write<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let mut file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)?;
    serde_json::to_writer(&mut file, value)?;
    file.sync_all()?;
    Ok(())
}
fn name(s: &str) -> Result<&str> {
    if s.is_empty()
        || !s
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
    {
        return Err("invalid local artifact name".into());
    }
    Ok(s)
}
fn raw_accuracy(m: &Value) -> Value {
    m.get("native_token_accuracy")
        .or_else(|| m.get("token_accuracy"))
        .cloned()
        .unwrap_or(Value::Null)
}
fn correct(m: &Value) -> Result<u64> {
    m.get("native_correct")
        .or_else(|| m.get("correct"))
        .and_then(Value::as_u64)
        .ok_or_else(|| "missing raw count".into())
}
fn policy(mode: &str) -> Result<fresh::Policy> {
    let wanted = match mode {
        "full_batch" => fresh::Mode::Full,
        "minibatch" => fresh::Mode::Mini,
        "forward_only" => fresh::Mode::Forward,
        _ => return Err("unknown original mode".into()),
    };
    fresh::training::policies()?
        .into_iter()
        .find(|p| p.mode == wanted)
        .ok_or_else(|| "missing reviewed331 mode".into())
}

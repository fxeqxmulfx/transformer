//! Original-data harness for CM19, using the unchanged data and metric ports.
use crate::{
    search::{self, Record},
    *,
};
use convex_stress_relabel::cohort::{number, write_new};
use convex_stress_relabel::{data, hash, ideal_loss, metrics, random::Random};
use general::{Mode, Model, Shape, Solver, State};
use propagation::{Observation, Problem, Verified};
use serde_json::{Value, json};
use std::{fs, path::Path};
mod audit;
mod fingerprint;
mod input;
mod replay;
mod run;
mod summary;
pub use fingerprint::fingerprint;
use input::*;
pub use replay::replay;
pub use run::run;
pub use summary::summarize;
#[cfg(test)]
mod tests;

const PROTOCOL: &str = include_str!("../../protocol.json");
const SUITE: &str = include_str!("../../../convex_stress/suite.json");
pub fn protocol() -> Result<Value> {
    Ok(serde_json::from_str(PROTOCOL)?)
}
pub(super) fn count(v: &Value, key: &str) -> Result<usize> {
    usize::try_from(
        v[key]
            .as_u64()
            .ok_or_else(|| format!("missing count {key}"))?,
    )
    .map_err(Into::into)
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Case {
    pub task: String,
    pub arm: String,
}
impl Case {
    pub fn id(&self) -> String {
        format!("{}--{}", self.arm, self.task)
    }
}
pub fn cases() -> Result<Vec<Case>> {
    let p = protocol()?;
    let mut out = vec![];
    for task in p["tasks"].as_array().ok_or("missing tasks")? {
        for arm in p["arms"].as_array().ok_or("missing modes")? {
            out.push(Case {
                task: task.as_str().ok_or("invalid task")?.into(),
                arm: arm.as_str().ok_or("invalid mode")?.into(),
            });
        }
    }
    Ok(out)
}
fn case(id: &str) -> Result<Case> {
    cases()?
        .into_iter()
        .find(|c| c.id() == id)
        .ok_or_else(|| "unplanned case".into())
}
fn mode(c: &Case) -> Result<Mode> {
    match c.arm.as_str() {
        "full_batch" => Ok(Mode::Full),
        "minibatch" => Ok(Mode::Mini),
        "forward_only" => Ok(Mode::Forward),
        _ => Err("invalid mode".into()),
    }
}
fn bytes(path: &Path, limit: u64) -> Result<Vec<u8>> {
    if fs::metadata(path)?.len() > limit {
        return Err("oversize saved artifact".into());
    }
    Ok(fs::read(path)?)
}
fn read(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&bytes(path, 64 * 1024 * 1024)?)?)
}
fn inventory(root: &Path) -> Result<Value> {
    let inv = read(&root.join("inventory.json"))?;
    if inv["source_fingerprint"] != fingerprint()
        || inv["protocol"] != protocol()?
        || inv["cases"] != serde_json::to_value(cases()?)?
    {
        return Err("inventory source/protocol mismatch".into());
    }
    Ok(inv)
}
pub fn prepare(root: &Path) -> Result<()> {
    if root.exists() {
        return Err("cohort destination already exists".into());
    }
    let producer = std::env::var("GENERAL_BANK_PRODUCER_COMMIT")?;
    if producer.len() != 40 || !producer.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("invalid producer commit".into());
    }
    if protocol()?["shared_fingerprint"] != convex_stress_relabel::cohort::fingerprint() {
        return Err("frozen shared sources have changed".into());
    }
    let mut tasks = vec![];
    for c in cases()?.iter().filter(|c| c.arm == "full_batch") {
        tasks.push(task_record(c, &inputs(c)?));
    }
    fs::create_dir(root)?;
    write_new(
        &root.join("inventory.json"),
        &json!({"source_fingerprint":fingerprint(),"producer_commit":producer,
        "protocol":protocol()?,"cases":cases()?,"tasks":tasks}),
    )
}
fn check_task(inv: &Value, c: &Case, x: &Inputs) -> Result<()> {
    let tasks = inv["tasks"].as_array().ok_or("missing task inventory")?;
    if tasks.iter().filter(|r| r["task"] == c.task).count() != 1
        || tasks.iter().find(|r| r["task"] == c.task) != Some(&task_record(c, x))
    {
        return Err("task differs from inventory".into());
    }
    Ok(())
}
fn metadata(c: &Case, inv: &Value, x: &Inputs) -> Result<Value> {
    Ok(
        json!({"case":c,"id":c.id(),"source_fingerprint":fingerprint(),"producer_commit":inv["producer_commit"],
        "protocol":protocol()?,"task_record":task_record(c,x),"schedule":schedule(c,x)?}),
    )
}
fn candidate_refusal(verified: Option<&Verified>, index: usize) -> Option<&'static str> {
    let v = match verified {
        Some(v) => v,
        None => return Some("no_complete_certificate"),
    };
    let cert = v.certificate();
    if !cert.branches.iter().any(|b| b.decoder == index && b.live) {
        return Some("no_live_branch");
    }
    if cert
        .problem
        .observations
        .iter()
        .any(|t| !cert.problem.decoders[index].contains(&t.label))
    {
        return Some("observed_class_outside_image");
    }
    None
}
fn choose(candidates: &[Value], vocab: usize) -> Result<Option<usize>> {
    let valid: Vec<_> = candidates
        .iter()
        .enumerate()
        .filter(|(_, r)| r["status"] == "complete")
        .collect();
    if valid.is_empty() {
        return Ok(None);
    }
    let losses = valid
        .iter()
        .map(|(_, r)| number(r, "train_ce"))
        .collect::<Result<Vec<_>>>()?;
    let k = convex_stress_relabel::choose_by_train(
        &losses,
        ideal_loss(vocab),
        number(&protocol()?, "ideal_ce_tolerance")?,
    )?;
    Ok(Some(valid[k].0))
}

//! Frozen CM15 categorical proposal inventory and unchanged original data.
use crate::{Model, Result, Sample, memory_bound, search};
use convex_stress_relabel::cohort::{number, write_new};
use convex_stress_relabel::{check_layout, data, hash, ideal_loss};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{fs, path::Path};
mod replay;
mod run;
mod summary;
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
pub fn fingerprint() -> String {
    let parent = convex_phase_start::cohort::fingerprint();
    let mut bytes = vec![];
    for source in [
        include_bytes!("../lib.rs").as_slice(),
        include_bytes!("../main.rs"),
        include_bytes!("../search.rs"),
        include_bytes!("../assignment.rs"),
        include_bytes!("mod.rs"),
        include_bytes!("run.rs"),
        include_bytes!("replay.rs"),
        include_bytes!("summary.rs"),
        include_bytes!("../../Cargo.toml"),
        include_bytes!("../../Cargo.lock"),
        include_bytes!("../../run_cohort.sh"),
        include_bytes!("../../../convex_stress_hull/limited.sh"),
        SUITE.as_bytes(),
        PROTOCOL.as_bytes(),
        parent.as_bytes(),
    ] {
        bytes.extend_from_slice(&(source.len() as u64).to_le_bytes());
        bytes.extend_from_slice(source);
    }
    hash(&bytes)
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Case {
    pub task: String,
    pub method: String,
}
impl Case {
    pub fn id(&self) -> String {
        format!("{}--{}", self.method, self.task)
    }
}
pub fn cases() -> Result<Vec<Case>> {
    let p = protocol()?;
    Ok(p["tasks"]
        .as_array()
        .ok_or("missing tasks")?
        .iter()
        .map(|t| Case {
            task: t.as_str().unwrap().into(),
            method: p["method"].as_str().unwrap().into(),
        })
        .collect())
}
fn case(id: &str) -> Result<Case> {
    cases()?
        .into_iter()
        .find(|c| c.id() == id)
        .ok_or_else(|| "unplanned decoder case".into())
}
struct Inputs {
    task: data::Task,
    row: Value,
    observations: Vec<Sample>,
    explicit_bytes: u64,
}
fn inputs(c: &Case) -> Result<Inputs> {
    if !cases()?.contains(c) {
        return Err("unplanned decoder case".into());
    }
    let suite: Value = serde_json::from_str(SUITE)?;
    let row = suite["tasks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["name"] == c.task)
        .ok_or("missing original task")?
        .clone();
    let task = data::build(&row)?;
    let explicit_bytes = memory_bound(task.vocab, task.vocab, task.vocab)?;
    check_layout(&task)?;
    let observations: Vec<_> = task.splits["train"]
        .iter()
        .map(|e| Sample {
            cell: e.tokens[1] as usize + task.vocab * e.tokens[0] as usize,
            label: e.targets[1] as usize,
        })
        .collect();
    crate::validate_observations(task.vocab, task.vocab, &observations)?;
    Ok(Inputs {
        task,
        row,
        observations,
        explicit_bytes,
    })
}
fn task_record(c: &Case, x: &Inputs) -> Value {
    json!({"task":c.task,"vocab":x.task.vocab,"training_examples":x.observations.len(),
        "data_fingerprints":x.row["data_fingerprints"],"selection":x.task.selection,
        "explicit_bytes_upper_bound":x.explicit_bytes})
}
fn check_task(inv: &Value, c: &Case, x: &Inputs) -> Result<()> {
    let tasks = inv["tasks"].as_array().ok_or("missing task inventory")?;
    if tasks.iter().filter(|t| t["task"] == c.task).count() != 1
        || tasks.iter().find(|t| t["task"] == c.task) != Some(&task_record(c, x))
    {
        return Err("original decoder task differs from inventory".into());
    }
    Ok(())
}
fn bytes(path: &Path, limit: u64) -> Result<Vec<u8>> {
    if fs::metadata(path)?.len() > limit {
        return Err("oversize saved decoder artifact".into());
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
        return Err("decoder inventory source/protocol mismatch".into());
    }
    Ok(inv)
}
fn parent_summary(parent: &Path) -> Result<Value> {
    if convex_phase_start::cohort::fingerprint() != protocol()?["parent_fingerprint"] {
        return Err("phase parent fingerprint changed".into());
    }
    let saved = read(&parent.join("summary.json"))?;
    let rebuilt = convex_phase_start::cohort::summarize(parent)?;
    if saved != rebuilt
        || saved["cells"] != 16
        || saved["choices"].as_array().map(Vec::len) != Some(24)
    {
        return Err("complete unchanged phase/cold/graph parents required".into());
    }
    Ok(saved)
}
pub fn prepare(root: &Path, parent: &Path) -> Result<()> {
    if root.exists() {
        return Err("decoder destination already exists".into());
    }
    let previous = parent_summary(parent)?;
    let producer = std::env::var("DECODE_PRODUCER_COMMIT")?;
    if producer.len() != 40 || !producer.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("invalid decoder producer commit".into());
    }
    let mut tasks = vec![];
    for c in cases()? {
        tasks.push(task_record(&c, &inputs(&c)?));
    }
    let parents = json!({"phase_root":fs::canonicalize(parent)?,
        "phase_summary_sha256":hash(&bytes(&parent.join("summary.json"),64*1024*1024)?)});
    fs::create_dir(root)?;
    write_new(&root.join("parent-summary.json"), &previous)?;
    write_new(
        &root.join("inventory.json"),
        &json!({"source_fingerprint":fingerprint(),
        "protocol":protocol()?,"cases":cases()?,"producer_commit":producer,"parents":parents,"tasks":tasks}),
    )
}
fn construct(x: &Inputs) -> Result<search::Proposal> {
    search::propose(
        x.task.vocab,
        x.task.vocab,
        &x.observations,
        serde_json::from_value(protocol()?["options"].clone())?,
    )
}
fn metrics_all(model: &Model, task: &data::Task) -> Result<Value> {
    let mut values = serde_json::Map::new();
    for (name, examples) in &task.splits {
        values.insert(name.clone(), crate::metrics::evaluate(model, examples)?);
    }
    Ok(Value::Object(values))
}
fn metadata(c: &Case, inv: &Value, x: &Inputs) -> Value {
    json!({"case":c,"id":c.id(),"source_fingerprint":fingerprint(),"protocol":protocol().unwrap(),
        "producer_commit":inv["producer_commit"],"parents":inv["parents"],"vocab":x.task.vocab,
        "data_fingerprints":x.row["data_fingerprints"],"selection":x.task.selection,
        "training_examples":x.observations.len(),"explicit_bytes_upper_bound":x.explicit_bytes,
        "data_validation_examples":x.observations.len(),"updates":0,"training_gradient_evaluations":0,
        "optimizer_examples_presented":0,"categorical_proposal_search":true})
}

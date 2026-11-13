//! Original tasks and immutable complete parents surround observed-only discovery.
use convex_stress_modular::{Result, coding, search};
use convex_stress_relabel::{data, hash, metrics};
use serde_json::{Value, json};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
};
mod fingerprint;
mod parent;
mod run;
mod summary;
pub use fingerprint::fingerprint;
pub use run::{replay, run};
pub use summary::summarize;
#[cfg(test)]
mod tests;
const PROTOCOL: &str = include_str!("../../protocol.json");
const SUITE: &str = include_str!("../../../../convex_stress/suite.json");
const LIMIT: u64 = 512 * 1024 * 1024;
fn protocol() -> Result<Value> {
    Ok(serde_json::from_str(PROTOCOL)?)
}
fn tasks() -> Result<Vec<String>> {
    serde_json::from_value(protocol()?["tasks"].clone()).map_err(Into::into)
}
fn modes() -> Result<Vec<String>> {
    serde_json::from_value(protocol()?["reference_modes"].clone()).map_err(Into::into)
}
fn bytes(path: &Path) -> Result<Vec<u8>> {
    if fs::metadata(path)?.len() > LIMIT {
        return Err("modular cohort artifact exceeds512MiB".into());
    }
    Ok(fs::read(path)?)
}
fn read<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    Ok(serde_json::from_slice(&bytes(path)?)?)
}
fn write<T: serde::Serialize>(path: &Path, value: &T) -> Result<()> {
    let raw = serde_json::to_vec(value)?;
    if raw.len() as u64 > LIMIT {
        return Err("modular artifact serialization exceeds512MiB".into());
    }
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?
        .write_all(&raw)?;
    Ok(())
}
fn binary_hash() -> Result<String> {
    Ok(hash(&bytes(&std::env::current_exe()?)?))
}
struct Inputs {
    task: data::Task,
    input: search::Input,
    record: Value,
}
fn assemble(task: data::Task, name: &str, data_fingerprints: Value) -> Result<Inputs> {
    convex_stress_relabel::check_layout(&task)?;
    let p = protocol()?;
    let input = search::Input {
        categorical: coding::Input {
            n: task.vocab,
            vocab: task.vocab,
            observations: task.splits["train"]
                .iter()
                .map(|e| {
                    [
                        e.tokens[1] as usize + task.vocab * e.tokens[0] as usize,
                        e.targets[1] as usize,
                    ]
                })
                .collect(),
            policy: serde_json::from_value(p["coding"].clone())?,
        },
        minimum_strip_support: p["minimum_strip_support"]
            .as_u64()
            .ok_or("missing strip support")? as usize,
        max_views: p["max_views"].as_u64().ok_or("missing strip view cap")? as usize,
    };
    let bound = input.validate()?;
    if bound > 48 * 1024 * 1024 * 1024 {
        return Err("modular aggregate bound exceeds48GiB".into());
    }
    let record = json!({"task":name,"vocab":task.vocab,"selection":task.selection,"data_fingerprints":data_fingerprints,
        "training_input_sha256":hash(&serde_json::to_vec(&input)?),"buffer_preflight_bytes":bound});
    Ok(Inputs {
        task,
        input,
        record,
    })
}
fn inputs(name: &str) -> Result<Inputs> {
    if !tasks()?.iter().any(|t| t == name) {
        return Err("unplanned modular task".into());
    }
    let suite: Value = serde_json::from_str(SUITE)?;
    let row = suite["tasks"]
        .as_array()
        .ok_or("missing task suite")?
        .iter()
        .find(|r| r["name"] == name)
        .ok_or("unknown original task")?;
    assemble(data::build(row)?, name, row["data_fingerprints"].clone())
}
fn inventory(root: &Path) -> Result<(Value, Value)> {
    let inv: Value = read(&root.join("inventory.json"))?;
    if inv["source_fingerprint"] != fingerprint()
        || inv["binary_sha256"] != binary_hash()?
        || inv["protocol"] != protocol()?
        || inv["tasks"] != serde_json::to_value(tasks()?)?
    {
        return Err("modular inventory source/binary/protocol differs".into());
    }
    let raw = bytes(&root.join("parent.json"))?;
    if inv["parent_sha256"] != hash(&raw)
        || inv["parent_sha256"] != protocol()?["parent_summary_sha256"]
    {
        return Err("complete parent binding changed".into());
    }
    let parent: Value = serde_json::from_slice(&raw)?;
    parent::validate(&parent)?;
    Ok((inv, parent))
}
fn check_input(inv: &Value, name: &str, x: &Inputs) -> Result<()> {
    let records = inv["inputs"].as_array().ok_or("missing input inventory")?;
    if records.iter().filter(|r| r["task"] == name).count() != 1
        || records.iter().find(|r| r["task"] == name) != Some(&x.record)
    {
        return Err("original task/input fingerprint differs from inventory".into());
    }
    Ok(())
}
pub fn prepare(root: &Path, parent_path: &Path) -> Result<()> {
    if root.exists() {
        return Err("modular destination already exists".into());
    }
    let producer = std::env::var("MODULAR_PRODUCER_COMMIT")?;
    if producer.len() != 40 || !producer.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err("invalid committed modular producer".into());
    }
    if protocol()?["shared_fingerprint"] != convex_stress_relabel::cohort::fingerprint() {
        return Err("frozen original data/metric source changed".into());
    }
    let raw = bytes(parent_path)?;
    let expected = protocol()?["parent_summary_sha256"]
        .as_str()
        .ok_or("complete parent is not pinned yet")?
        .to_string();
    if hash(&raw) != expected {
        return Err("complete full105 parent differs from freeze".into());
    }
    let parent: Value = serde_json::from_slice(&raw)?;
    parent::validate(&parent)?;
    let records: Vec<_> = tasks()?
        .iter()
        .map(|name| {
            let x = inputs(name)?;
            parent::choices(&parent, name, &x.record)?;
            Ok(x.record)
        })
        .collect::<Result<_>>()?;
    fs::create_dir(root)?;
    OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(root.join("parent.json"))?
        .write_all(&raw)?;
    write(
        &root.join("inventory.json"),
        &json!({"source_fingerprint":fingerprint(),"binary_sha256":binary_hash()?,
        "producer_commit":producer,"protocol":protocol()?,"tasks":tasks()?,"inputs":records,"parent_sha256":expected}),
    )
}

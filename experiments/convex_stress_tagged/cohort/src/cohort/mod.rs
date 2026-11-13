//! Frozen original-task protocol; model constructors receive only training labels.
use convex_stress_density::{balanced_search, interval_search, refinement_search, search};
use convex_stress_relabel::{data, hash, metrics};
use convex_stress_tagged::{Model, Result, search as tagged};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
};
mod fingerprint;
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
    protocol()?["tasks"]
        .as_array()
        .ok_or("missing task list")?
        .iter()
        .map(|t| {
            t.as_str()
                .map(String::from)
                .ok_or_else(|| "invalid task name".into())
        })
        .collect()
}
fn bytes(path: &Path) -> Result<Vec<u8>> {
    if fs::metadata(path)?.len() > LIMIT {
        return Err("cohort artifact exceeds byte cap".into());
    }
    Ok(fs::read(path)?)
}
fn read<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    Ok(serde_json::from_slice(&bytes(path)?)?)
}
fn write<T: serde::Serialize>(path: &Path, value: &T) -> Result<()> {
    let raw = serde_json::to_vec(value)?;
    if raw.len() as u64 > LIMIT {
        return Err("cohort artifact exceeds byte cap".into());
    }
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?
        .write_all(&raw)?;
    Ok(())
}
struct Inputs {
    task: data::Task,
    record: Value,
    input: interval_search::Input,
}
fn assemble(task: data::Task, name: &str, fingerprints: Value) -> Result<Inputs> {
    convex_stress_relabel::check_layout(&task)?;
    let p = protocol()?;
    let input = interval_search::Input {
        categorical: search::Input {
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
            options: serde_json::from_value(p["exact"].clone())?,
        },
        policy: serde_json::from_value(p["interval"].clone())?,
    };
    search::validate(&input.categorical)?;
    let bound = p["buffer_preflight_bytes"]
        .as_u64()
        .ok_or("missing buffer bound")?;
    if bound > 64 * 1024 * 1024 * 1024 {
        return Err("preflight exceeds process cap".into());
    }
    let record = json!({"task":name,"vocab":task.vocab,"selection":task.selection,
        "data_fingerprints":fingerprints,"observations":input.categorical.observations.len(),
        "training_input_sha256":hash(&serde_json::to_vec(&input)?),"buffer_preflight_bytes":bound});
    Ok(Inputs {
        task,
        record,
        input,
    })
}
fn inputs(name: &str) -> Result<Inputs> {
    if !tasks()?.iter().any(|t| t == name) {
        return Err("unplanned density task".into());
    }
    let suite: Value = serde_json::from_str(SUITE)?;
    let row = suite["tasks"]
        .as_array()
        .ok_or("missing suite")?
        .iter()
        .find(|t| t["name"] == name)
        .ok_or("missing original task")?;
    assemble(data::build(row)?, name, row["data_fingerprints"].clone())
}
fn inventory(root: &Path) -> Result<Value> {
    let inv: Value = read(&root.join("inventory.json"))?;
    if inv["source_fingerprint"] != fingerprint()
        || inv["binary_sha256"] != binary_fingerprint()?
        || inv["protocol"] != protocol()?
        || inv["tasks"] != serde_json::to_value(tasks()?)?
    {
        return Err("density inventory source/protocol mismatch".into());
    }
    Ok(inv)
}
fn binary_fingerprint() -> Result<String> {
    Ok(hash(&bytes(&std::env::current_exe()?)?))
}
fn check_record(inv: &Value, name: &str, x: &Inputs) -> Result<()> {
    let records = inv["inputs"].as_array().ok_or("missing input inventory")?;
    if records.iter().filter(|r| r["task"] == name).count() != 1
        || records.iter().find(|r| r["task"] == name) != Some(&x.record)
    {
        return Err("original input differs from frozen inventory".into());
    }
    Ok(())
}
pub fn prepare(root: &Path) -> Result<()> {
    if root.exists() {
        return Err("density cohort destination exists".into());
    }
    let producer = std::env::var("TAGGED_PRODUCER_COMMIT")?;
    if producer.len() != 40 || !producer.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("invalid committed density producer".into());
    }
    if protocol()?["shared_fingerprint"] != convex_stress_relabel::cohort::fingerprint() {
        return Err("frozen shared data/metric source changed".into());
    }
    let records: Vec<_> = tasks()?
        .iter()
        .map(|t| Ok(inputs(t)?.record))
        .collect::<Result<_>>()?;
    fs::create_dir(root)?;
    write(
        &root.join("inventory.json"),
        &json!({"source_fingerprint":fingerprint(),"binary_sha256":binary_fingerprint()?,
        "producer_commit":producer,"protocol":protocol()?,"tasks":tasks()?,"inputs":records}),
    )
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Report {
    parent: balanced_search::Report,
    extension: tagged::Report,
    selected: Option<String>,
    buffer_bound_bytes: u64,
}
fn construct(input: &interval_search::Input) -> Result<(Value, Report)> {
    let samples = search::validate(&input.categorical)?;
    let old = convex_stress_decode::search::propose(
        input.categorical.n,
        input.categorical.vocab,
        &samples,
        serde_json::from_value(protocol()?["earlier_decode"].clone())?,
    )?;
    let prior: Vec<_> = old.model.iter().cloned().collect();
    let out = interval_search::propose(input.clone())?;
    let parent = interval_search::compare(out, prior.clone())?;
    let refined = refinement_search::run(
        input,
        &prior,
        parent,
        serde_json::from_value(protocol()?["refinement"].clone())?,
    )?;
    let balanced = balanced_search::run(
        input,
        &prior,
        refined,
        serde_json::from_value(protocol()?["balanced"].clone())?,
    )?;
    let extension = tagged::run(tagged::Input {
        n: input.categorical.n,
        vocab: input.categorical.vocab,
        observations: input.categorical.observations.clone(),
        policy: serde_json::from_value(protocol()?["tagged"].clone())?,
    })?;
    let bound = balanced.buffer_bound_bytes + extension.buffer_bound_bytes + 512 * 1024 * 1024;
    if bound > 48 * 1024 * 1024 * 1024 {
        return Err("retained-parent/tagged aggregate exceeds48GiB forecast".into());
    }
    let mut report = Report {
        parent: balanced,
        extension,
        selected: None,
        buffer_bound_bytes: bound,
    };
    let mut best = f64::INFINITY;
    for (id, _, ce) in scored_candidates(&report)? {
        if ce + 1e-12 < best {
            best = ce;
            report.selected = Some(id);
        }
    }
    Ok((serde_json::to_value(old)?, report))
}
fn scored_candidates(r: &Report) -> Result<Vec<(String, Model, f64)>> {
    let mut all = vec![];
    for (id, m, ce) in balanced_search::candidates(&r.parent)? {
        m.validate_shape()?;
        let model = Model::construct(
            m.vocab,
            m.row_phases.clone(),
            m.column_phases.clone(),
            m.decoder.clone(),
        )?;
        if model.latent_labels != m.latent_labels || model.output_labels != m.output_labels {
            return Err("old actual table changed during model conversion".into());
        }
        all.push((id, model, ce));
    }
    for (i, attempt) in r.extension.attempts.iter().enumerate() {
        if let Some(model) = &attempt.model {
            let ce = attempt.audit.as_ref().ok_or("missing tagged audit")?["train_ce"]
                .as_f64()
                .ok_or("missing tagged CE")?;
            all.push((format!("tagged-{i}"), model.clone(), ce));
        }
    }
    Ok(all)
}
fn candidates(r: &Report) -> Result<Vec<(String, Model)>> {
    Ok(scored_candidates(r)?
        .into_iter()
        .map(|(id, m, _)| (id, m))
        .collect())
}
fn selected(r: &Report) -> Option<String> {
    r.selected.clone()
}

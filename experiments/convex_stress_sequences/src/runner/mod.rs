//! Fresh immutable training states precede held-out scoring. Replays reconstruct
//! discovery, actual full-head updates, masks, predictions and every refusal.
use crate::{CATALOG, PROTOCOL, Result, adapter, complex::binding, dataset, hash};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{fs, path::Path, process::Command};
pub mod artifacts;
mod score;
use artifacts::{digest, read, write};
pub use score::{measurements, replay, score, summary};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Stage {
    pub dataset: Value,
    pub request: binding::Request,
    pub model: Option<binding::Report>,
    pub error: Option<String>,
}

fn catalog() -> Result<Value> {
    Ok(serde_json::from_str(CATALOG)?)
}
pub fn cases() -> Result<Vec<String>> {
    Ok(serde_json::from_value(catalog()?["first_cohort"].clone())?)
}

fn binary() -> Result<String> {
    digest(&std::env::current_exe()?)
}
fn valid_commit(commit: &str) -> bool {
    commit.len() == 40 && commit.bytes().all(|c| c.is_ascii_hexdigit())
}

pub fn prepare(root: &Path, producer: &str, history: &Path) -> Result<()> {
    if root.exists() || !valid_commit(producer) {
        return Err("fresh root and full commit required".into());
    }
    let head = Command::new("git").args(["rev-parse", "HEAD"]).output()?;
    let status = Command::new("git")
        .args(["status", "--porcelain"])
        .output()?;
    if !head.status.success()
        || !status.status.success()
        || String::from_utf8(head.stdout)?.trim() != producer
        || !status.stdout.is_empty()
    {
        return Err("prepare from the clean committed producer checkout".into());
    }
    let cat = catalog()?;
    if digest(history)? != cat["input_sha256"]["history"] {
        return Err("historical GPT snapshot bytes differ".into());
    }
    let history_bytes = fs::read(history)?;
    let inputs = cases()?
        .iter()
        .map(|c| dataset::build(c).map(|d| d.metadata()))
        .collect::<Result<Vec<_>>>()?;
    let inv = json!({"producer_commit":producer,"binary_sha256":binary()?,
        "protocol_sha256":hash(PROTOCOL.as_bytes()),"catalog_sha256":hash(CATALOG.as_bytes()),
        "protocol":serde_json::from_str::<Value>(PROTOCOL)?,"inputs":inputs,
        "history_sha256":cat["input_sha256"]["history"],"cases":cases()?});
    fs::create_dir(root)?;
    let mut history_file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(root.join("history.json"))?;
    use std::io::Write;
    history_file.write_all(&history_bytes)?;
    history_file.sync_all()?;
    write(&root.join("inventory.json"), &inv)
}

fn inventory(root: &Path) -> Result<Value> {
    let inv: Value = read(&root.join("inventory.json"))?;
    let inputs = cases()?
        .iter()
        .map(|c| dataset::build(c).map(|d| d.metadata()))
        .collect::<Result<Vec<_>>>()?;
    if !inv["producer_commit"].as_str().is_some_and(valid_commit)
        || inv["binary_sha256"] != binary()?
        || inv["protocol_sha256"] != hash(PROTOCOL.as_bytes())
        || inv["catalog_sha256"] != hash(CATALOG.as_bytes())
        || inv["protocol"] != serde_json::from_str::<Value>(PROTOCOL)?
        || inv["inputs"] != json!(inputs)
        || inv["cases"] != json!(cases()?)
        || inv["history_sha256"] != catalog()?["input_sha256"]["history"]
        || inv["history_sha256"] != digest(&root.join("history.json"))?
    {
        return Err("source, binary, protocol, data or history inventory changed".into());
    }
    Ok(inv)
}

pub fn construct(
    vocabulary: usize,
    rows: &[crate::data::Example],
    metadata: Value,
) -> Result<Stage> {
    let request = adapter::request(vocabulary, rows)?;
    let (model, error) = match binding::run(request.clone()) {
        Ok(model) => (Some(model), None),
        Err(error) => (None, Some(error.to_string())),
    };
    Ok(Stage {
        dataset: metadata,
        request,
        model,
        error,
    })
}

pub fn stage(root: &Path, case: &str) -> Result<()> {
    inventory(root)?;
    let data = dataset::build(case)?;
    let dir = root.join(case);
    fs::create_dir(&dir)?;
    write(
        &dir.join("started.json"),
        &json!({"case":case,"phase":"training_only"}),
    )?;
    let stage = construct(data.task.vocab, &data.task.splits["train"], data.metadata())?;
    write(&dir.join("state.json"), &stage)?;
    write(
        &dir.join("seal.json"),
        &json!({"state_sha256":digest(&dir.join("state.json"))?,
        "inventory_sha256":digest(&root.join("inventory.json"))?,"heldout_scoring_completed":false}),
    )
}

fn load(root: &Path, case: &str) -> Result<(dataset::Dataset, Stage, Value)> {
    inventory(root)?;
    let data = dataset::build(case)?;
    let dir = root.join(case);
    let seal: Value = read(&dir.join("seal.json"))?;
    if seal
        != json!({"state_sha256":digest(&dir.join("state.json"))?,
        "inventory_sha256":digest(&root.join("inventory.json"))?,"heldout_scoring_completed":false})
    {
        return Err("training state seal differs".into());
    }
    let stage: Stage = read(&dir.join("state.json"))?;
    if stage.dataset != data.metadata()
        || stage.request != adapter::request(data.task.vocab, &data.task.splits["train"])?
        || stage.model.is_some() == stage.error.is_some()
        || stage
            .model
            .as_ref()
            .is_some_and(|m| m.request != stage.request)
    {
        return Err("sealed training request or data differs".into());
    }
    Ok((data, stage, seal))
}

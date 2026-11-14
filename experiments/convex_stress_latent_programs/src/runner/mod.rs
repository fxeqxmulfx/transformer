//! Freeze source and baseline; seal every training state before any held-out
//! score. No dataset identity is passed into discovery, the head or training.
use crate::{Input, PROTOCOL, Result, binding, dataset, hash, sequences};
use artifacts::{digest, read, write};
pub use sequences::runner::{artifacts, cases};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{fs, path::Path, process::Command};
mod score;
pub use score::{measurements, replay, score, summary};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Stage {
    pub dataset: Value,
    pub input: Input,
    pub model: Option<binding::Report>,
    pub error: Option<String>,
}
fn binary() -> Result<String> {
    digest(&std::env::current_exe()?)
}
fn valid_commit(commit: &str) -> bool {
    commit.len() == 40 && commit.bytes().all(|c| c.is_ascii_hexdigit())
}
fn protocol() -> Result<Value> {
    Ok(serde_json::from_str(PROTOCOL)?)
}
fn inputs() -> Result<Vec<Value>> {
    cases()?
        .iter()
        .map(|c| dataset::build(c).map(|d| d.metadata()))
        .collect()
}
pub fn prepare(root: &Path, producer: &str, baseline: &Path) -> Result<()> {
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
    let p = protocol()?;
    if digest(baseline)? != p["baseline_report_sha256"] {
        return Err("frozen task322 baseline bytes differ".into());
    }
    let inv = json!({"producer_commit":producer,"binary_sha256":binary()?,
        "protocol_sha256":hash(PROTOCOL.as_bytes()),"protocol":p,
        "catalog_sha256":hash(sequences::CATALOG.as_bytes()),"inputs":inputs()?,"cases":cases()?,
        "baseline_sha256":digest(baseline)?});
    let bytes = fs::read(baseline)?;
    fs::create_dir(root)?;
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(root.join("baseline.json"))?;
    use std::io::Write;
    file.write_all(&bytes)?;
    file.sync_all()?;
    write(&root.join("inventory.json"), &inv)
}
fn inventory(root: &Path) -> Result<Value> {
    let inv: Value = read(&root.join("inventory.json"))?;
    if !inv["producer_commit"].as_str().is_some_and(valid_commit)
        || inv["binary_sha256"] != binary()?
        || inv["protocol_sha256"] != hash(PROTOCOL.as_bytes())
        || inv["protocol"] != protocol()?
        || inv["catalog_sha256"] != hash(sequences::CATALOG.as_bytes())
        || inv["inputs"] != json!(inputs()?)
        || inv["cases"] != json!(cases()?)
        || inv["baseline_sha256"] != protocol()?["baseline_report_sha256"]
        || inv["baseline_sha256"] != digest(&root.join("baseline.json"))?
    {
        return Err("source, binary, protocol, baseline or data inventory changed".into());
    }
    Ok(inv)
}
pub fn construct(
    vocabulary: usize,
    rows: &[crate::data::Example],
    metadata: Value,
) -> Result<Stage> {
    let input = sequences::adapter::request(vocabulary, rows)?
        .discovery
        .input;
    let (model, error) = match binding::run(input.clone()) {
        Ok(model) => (Some(model), None),
        Err(e) => (None, Some(e.to_string())),
    };
    Ok(Stage {
        dataset: metadata,
        input,
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
        || stage.input
            != sequences::adapter::request(data.task.vocab, &data.task.splits["train"])?
                .discovery
                .input
        || stage.model.is_some() == stage.error.is_some()
        || stage.model.as_ref().is_some_and(|m| m.input != stage.input)
    {
        return Err("sealed training input or data differs".into());
    }
    Ok((data, stage, seal))
}
fn all_sealed(root: &Path) -> Result<()> {
    for case in cases()? {
        load(root, &case)?;
    }
    Ok(())
}
fn baseline_case(root: &Path, case: &str) -> Result<Value> {
    let baseline: Value = read(&root.join("baseline.json"))?;
    let rows: Vec<_> = baseline["cases"]
        .as_array()
        .ok_or("missing baseline cases")?
        .iter()
        .filter(|r| r["results"]["case"] == case)
        .cloned()
        .collect();
    if rows.len() != 1 {
        return Err("baseline case missing or duplicated".into());
    }
    Ok(rows[0].clone())
}

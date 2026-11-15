//! Immutable source/data/binary binding, complete staging before scoring and
//! fresh reconstruction on every score/replay. Case names stay in this adapter.
use crate::{
    discovery,
    model::{self, Prior, Trained},
    training::Policy,
    *,
};
use artifacts::{digest, read, write};
pub use parent::sequences::runner::artifacts;
use serde_json::{Value, json};
use std::{fs, path::Path, process::Command};
mod score;
#[cfg(test)]
mod tests;
pub use score::{replay, score, summary};

pub fn protocol() -> Result<Value> {
    Ok(serde_json::from_str(PROTOCOL)?)
}
pub fn policies() -> Result<Vec<Policy>> {
    Ok(serde_json::from_value(protocol()?["policies"].clone())?)
}
fn coordinates(v: usize) -> Vec<f64> {
    (0..v).map(|c| (c as f64 - 100.) / 128.).collect()
}
fn cases() -> Vec<String> {
    frequency::CASES.iter().map(|s| s.to_string()).collect()
}
fn binary() -> Result<String> {
    digest(&std::env::current_exe()?)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Endpoint {
    pub prior: Prior,
    pub policy: Policy,
    pub trained: Option<Trained>,
    pub error: Option<String>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Stage {
    pub metadata: Value,
    pub input: Input,
    pub discovery: Option<discovery::Report>,
    pub endpoints: Vec<Endpoint>,
    pub complete_case_forecast: Value,
    pub error: Option<String>,
}

fn data_case(inv: &Value, case: &str) -> Result<Value> {
    let rows = inv["data_cases"].as_array().ok_or("missing data cases")?;
    let found: Vec<_> = rows.iter().filter(|r| r["case"] == case).collect();
    if found.len() != 1 {
        return Err("missing or duplicated fixed case".into());
    }
    Ok(found[0].clone())
}

pub fn prepare(root: &Path, producer: &str, data: &Path, original: &Path) -> Result<()> {
    if root.exists() || producer.len() != 40 || !producer.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err("fresh root and full producer commit required".into());
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
        return Err("clean committed producer required".into());
    }
    let p = protocol()?;
    let data_bytes = fs::metadata(data)?.len();
    let original_bytes = fs::metadata(original)?.len();
    let buffer = 64u128 * data_bytes as u128 + (4u128 << 30) + original_bytes as u128;
    if buffer > 56 * 1024u128.pow(3) || original_bytes > 1 << 30 || data_bytes > 1 << 30 {
        return Err("complete reference parse/copy forecast refused".into());
    }
    if digest(data)? != p["data_inventory_sha256"]
        || digest(original)? != p["original35_summary_sha256"]
    {
        return Err("reviewed frequency data or original35 reference differs".into());
    }
    let source: Value = read(data)?;
    if source["baseline334"]["complete"] != true
        || source["baseline334"]["cohort_cases"] != 16
        || source["cases"]
            .as_array()
            .ok_or("missing seven cases")?
            .len()
            != 7
    {
        return Err("incomplete frequency inventory or controls".into());
    }
    let data_cases = source["cases"].clone();
    drop(source);
    let inv = json!({"producer_commit":producer,"binary_sha256":binary()?,"protocol":p,
        "protocol_sha256":hash(PROTOCOL.as_bytes()),"cases":cases(),"data_cases":data_cases,
        "data_bytes":data_bytes,"original35_bytes":original_bytes,"reference_buffer_bytes":buffer as u64});
    for case in cases() {
        let rebuilt = frequency::dataset::build(&case)?;
        if data_case(&inv, &case)?["data"] != frequency::dataset::metadata(&rebuilt) {
            return Err("regenerated original frequency metadata differs".into());
        }
    }
    fs::create_dir(root)?;
    fs::copy(data, root.join("reference-data.json"))?;
    fs::copy(original, root.join("original35.json"))?;
    write(&root.join("inventory.json"), &inv)
}

fn inventory(root: &Path) -> Result<Value> {
    let inv: Value = read(&root.join("inventory.json"))?;
    let p = protocol()?;
    if inv["binary_sha256"] != binary()?
        || inv["protocol"] != p
        || inv["protocol_sha256"] != hash(PROTOCOL.as_bytes())
        || inv["cases"] != json!(cases())
        || digest(&root.join("reference-data.json"))? != p["data_inventory_sha256"]
        || digest(&root.join("original35.json"))? != p["original35_summary_sha256"]
    {
        return Err("binary, protocol, complete references or cases changed".into());
    }
    Ok(inv)
}

pub fn repeat_run(source: &Path, root: &Path) -> Result<()> {
    if root.exists() {
        return Err("fresh independent repeat directory required".into());
    }
    inventory(source)?;
    fs::create_dir(root)?;
    for file in ["inventory.json", "reference-data.json", "original35.json"] {
        fs::copy(source.join(file), root.join(file))?;
    }
    Ok(())
}

pub fn construct(input: Input, metadata: Value, scoring: &evaluation::Forecast) -> Result<Stage> {
    if scoring.scalar_work > 4_000_000_000_000 || scoring.raw_bytes > 256 * 1024u64.pow(3) {
        return Err("invalid scoring preflight supplied to constructor".into());
    }
    let policies = policies()?;
    let mut endpoints = vec![];
    let mut error = None;
    let (search, search_buffer) = match discovery::forecast(&input, &coordinates(input.vocabulary))
    {
        Ok((work, bytes)) => (work as u128, bytes as u128),
        Err(e) => {
            error = Some(e.to_string());
            (0, 0)
        }
    };
    let mut training_work = 0u128;
    let mut buffers = 0u128;
    for p in &policies {
        let f = p.forecast(input.observations.len(), input.vocabulary)?;
        training_work += 2 * f.scalar_work as u128;
        buffers += 2 * f.buffer_bytes as u128;
    }
    // Stage, score-load and fresh replay each reconstruct the entire search
    // and six trajectories. Scores include both metric and audited products.
    let complete_work = 3 * (search + training_work) + 12 * scoring.scalar_work as u128;
    let complete_buffer = search_buffer + buffers + scoring.raw_bytes as u128 + (2u128 << 30);
    let forecast = json!({"search":search as u64,"all_six_training":training_work as u64,
        "one_endpoint_scoring":scoring,"complete_stage_score_replay_work":complete_work as u64,
        "complete_buffer_bytes":complete_buffer as u64,"construction_passes":3,
        "scoring_passes_per_endpoint":2,"process_address_space_bytes":68_719_476_736u64,
        "scope":"all candidate work, complete all-class products, six CE histories, score and fresh replay; hashing/serialization/data reconstruction measured additionally"});
    if complete_work > 600_000_000_000_000 || complete_buffer > 56 * 1024u128.pow(3) {
        error = Some("complete case work or memory forecast refused".into());
    }
    // The composed admission check precedes every candidate fit and CE step.
    let report = if error.is_none() {
        match discovery::run(input.clone(), coordinates(input.vocabulary)) {
            Ok(report) => Some(report),
            Err(failure) => {
                error = Some(failure.to_string());
                None
            }
        }
    } else {
        None
    };
    for prior in [Prior::Coordinates, Prior::Categorical] {
        for policy in &policies {
            let (trained, failure) = if error.is_some() {
                (None, error.clone())
            } else if let Some(report) = &report {
                match model::train(report, prior, policy.clone()) {
                    Ok(trained) => (Some(trained), None),
                    Err(e) => (None, Some(e.to_string())),
                }
            } else {
                (None, Some("missing candidate search".into()))
            };
            endpoints.push(Endpoint {
                prior,
                policy: policy.clone(),
                trained,
                error: failure,
            });
        }
    }
    Ok(Stage {
        metadata,
        input,
        discovery: report,
        endpoints,
        complete_case_forecast: forecast,
        error,
    })
}

fn dataset(inv: &Value, case: &str) -> Result<frequency::dataset::Dataset> {
    let data = frequency::dataset::build(case)?;
    if data_case(inv, case)?["data"] != frequency::dataset::metadata(&data) {
        return Err("original dataset metadata or vocabulary changed".into());
    }
    Ok(data)
}

pub fn stage(root: &Path, case: &str) -> Result<()> {
    let inv = inventory(root)?;
    let data = dataset(&inv, case)?;
    let input = parent::dataset::input(data.task.vocab, &data.task.splits["train"])?;
    let scoring = evaluation::forecast(data.task.vocab, &data.task.splits)?;
    let stage = construct(input, frequency::dataset::metadata(&data), &scoring)?;
    let dir = root.join(case);
    fs::create_dir(&dir)?;
    write(&dir.join("state.json"), &stage)?;
    write(
        &dir.join("seal.json"),
        &json!({"case":case,"state_sha256":digest(&dir.join("state.json"))?,
        "metadata_sha256":hash(&serde_json::to_vec(&stage.metadata)?),"endpoint_count":stage.endpoints.len(),
        "all_modes_and_priors_sealed_before_scoring":true}),
    )
}

fn all_sealed(root: &Path) -> Result<()> {
    for case in cases() {
        let seal: Value = read(&root.join(&case).join("seal.json"))?;
        if seal["case"] != case
            || seal["endpoint_count"] != 6
            || seal["all_modes_and_priors_sealed_before_scoring"] != true
            || seal["state_sha256"] != digest(&root.join(&case).join("state.json"))?
        {
            return Err("every case and mode must be sealed before scoring".into());
        }
    }
    Ok(())
}

fn load(
    root: &Path,
    inv: &Value,
    case: &str,
) -> Result<(frequency::dataset::Dataset, Stage, Value)> {
    let data = dataset(inv, case)?;
    let dir = root.join(case);
    let saved: Stage = read(&dir.join("state.json"))?;
    let seal: Value = read(&dir.join("seal.json"))?;
    if seal["state_sha256"] != digest(&dir.join("state.json"))? {
        return Err("training seal changed".into());
    }
    let input = parent::dataset::input(data.task.vocab, &data.task.splits["train"])?;
    let scoring = evaluation::forecast(data.task.vocab, &data.task.splits)?;
    let rebuilt = construct(input, frequency::dataset::metadata(&data), &scoring)?;
    if saved != rebuilt {
        return Err("complete candidate search or six CE states changed".into());
    }
    Ok((data, rebuilt, seal))
}

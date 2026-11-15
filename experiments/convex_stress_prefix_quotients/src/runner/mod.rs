//! Immutable producer/input binding and complete training seals before scores.
//! Adapted from task339's reviewed runner at34ea5801, with CM57 multi-program
//! forecasts, original prefix scoring and streaming reference authentication.
use crate::{
    artifacts::{digest, read, write},
    *,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{fs, path::Path, process::Command};
mod score;
mod summary;
pub use score::{replay, score};
pub use summary::summary;
#[cfg(test)]
mod tests;

pub fn protocol() -> Result<Value> {
    Ok(serde_json::from_str(PROTOCOL)?)
}
fn binary() -> Result<String> {
    digest(&std::env::current_exe()?)
}
fn cases() -> Vec<String> {
    dataset::CASES.iter().map(|s| s.to_string()).collect()
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Prepared {
    pub metadata: Value,
    pub training: Option<fit::Stage>,
    pub complete_forecast: Value,
    pub error: Option<String>,
}
fn data_case(inv: &Value, case: &str) -> Result<Value> {
    let cases = inv["data_cases"]
        .as_array()
        .ok_or("missing thirteen data cases")?;
    let matches: Vec<_> = cases.iter().filter(|r| r["case"] == case).collect();
    if matches.len() != 1 {
        return Err("missing or duplicated case".into());
    }
    Ok(matches[0]["data"].clone())
}
fn metadata(data: &dataset::dataset::Dataset) -> Result<Value> {
    from_data(dataset::dataset::metadata(data))
}
fn rebuild_data(inv: &Value, case: &str) -> Result<dataset::dataset::Dataset> {
    let data = from_data(dataset::dataset::build(case))?;
    if data_case(inv, case)? != metadata(&data)? {
        return Err("original dataset metadata changed".into());
    }
    Ok(data)
}

pub fn prepare(root: &Path, producer: &str, data: &Path, original: &Path) -> Result<()> {
    if root.exists() || producer.len() != 40 || !producer.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("fresh run directory and full producer commit required".into());
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
        return Err("clean committed complete producer required before measurement".into());
    }
    let p = protocol()?;
    let data_bytes = fs::metadata(data)?.len();
    let original_bytes = fs::metadata(original)?.len();
    let buffer = 64 * data_bytes as u128 + 4 * 1024u128.pow(3) + original_bytes as u128;
    if buffer > joint::MAX_BYTES || data_bytes > 1024u64.pow(3) || original_bytes > 1024u64.pow(3) {
        return Err("complete reference parse/copy forecast refused".into());
    }
    if digest(data)? != p["baseline343_report_sha256"]
        || digest(original)? != p["original35_summary_sha256"]
    {
        return Err("reviewed343 report or original35 reference differs".into());
    }
    let source: Value = read(data)?;
    if source["complete"] != true
        || source["new_cases"] != 13
        || source["measured_catalog_cases"] != 68
        || source["catalog_all_modes_all_heldout99_cases"] != 68
        || source["reference340"]["new_model_evaluations"] != 0
        || source["reference340"]["new_data_cases"] != 13
        || artifacts::canonical_digest(&source["reference340"])? != p["data_inventory_sha256"]
        || source["reference340"]["baseline339"]["complete"] != true
        || source["reference340"]["baseline339"]["cohort_cases"] != 23
        || artifacts::canonical_digest(&source["reference340"]["baseline339"])?
            != p["baseline339_report_sha256"]
    {
        return Err("incomplete343 history, data340 inventory or inherited339 report".into());
    }
    let data_cases = source["reference340"]["cases"].clone();
    drop(source);
    let inv = json!({"producer_commit":producer,"binary_sha256":binary()?,"protocol":p,
        "protocol_sha256":dataset::hash(PROTOCOL.as_bytes()),"cases":cases(),"data_cases":data_cases,
        "data_bytes":data_bytes,"original35_bytes":original_bytes,"reference_buffer_bytes":buffer as u64});
    for case in cases() {
        rebuild_data(&inv, &case)?;
    }
    fs::create_dir(root)?;
    fs::copy(data, root.join("baseline343.json"))?;
    fs::copy(original, root.join("original35.json"))?;
    write(&root.join("inventory.json"), &inv)
}

fn inventory(root: &Path) -> Result<Value> {
    let inv: Value = read(&root.join("inventory.json"))?;
    let p = protocol()?;
    if inv["binary_sha256"] != binary()?
        || inv["protocol"] != p
        || inv["protocol_sha256"] != dataset::hash(PROTOCOL.as_bytes())
        || inv["cases"] != json!(cases())
        || digest(&root.join("baseline343.json"))? != p["baseline343_report_sha256"]
        || digest(&root.join("original35.json"))? != p["original35_summary_sha256"]
    {
        return Err("binary, protocol or complete references changed".into());
    }
    Ok(inv)
}
pub fn repeat_run(source: &Path, root: &Path) -> Result<()> {
    if root.exists() {
        return Err("fresh repeat directory required".into());
    }
    inventory(source)?;
    fs::create_dir(root)?;
    for file in ["inventory.json", "baseline343.json", "original35.json"] {
        fs::copy(source.join(file), root.join(file))?;
    }
    Ok(())
}

fn preflight(
    input: &adapter::Input,
    settings: &fit::Settings,
    data: &dataset::dataset::Dataset,
) -> Result<Value> {
    let train = fit::forecast(input, settings)?;
    let scoring = evaluation::forecast(input.vocabulary, settings.max_programs, &data.task.splits)?;
    let work = 3 * train.scalar_work as u128 + 12 * scoring.scalar_work as u128;
    let buffer = train.buffer_bytes as u128 + scoring.buffer_bytes as u128 + 2 * 1024u128.pow(3);
    let disk = 12 * scoring.disk_bytes as u128;
    let complete_forecast = json!({"one_construction":train,"one_endpoint_evaluation":scoring,
            "construction_passes":3,"scoring_passes_per_endpoint":2,"endpoints":6,
            "complete_scalar_work":work as u64,"complete_buffer_bytes":buffer as u64,"raw_disk_bytes":disk as u64,
            "process_address_space_bytes":68719476736u64,
            "scope":"all candidate search, six training traces, two full scored product audits and fresh reconstruction; hashing, JSON and data regeneration measured additionally"});
    if work > 600_000_000_000_000 || buffer > joint::MAX_BYTES || disk > 256 * 1024u128.pow(3) {
        return Err("composed case work, memory or disk forecast refused".into());
    }
    Ok(complete_forecast)
}
/// Data shapes/masks and forecasts only: no program search, fitting or scores.
pub fn forecast_all(path: &Path) -> Result<()> {
    let mut forecasts = vec![];
    let settings = fit::settings()?;
    for case in cases() {
        let data = from_data(dataset::dataset::build(&case))?;
        let input = adapter::input(data.task.vocab, &data.task.splits["train"])?;
        forecasts.push(json!({"case":case,"forecast":preflight(&input,&settings,&data)?}));
    }
    write(
        path,
        &json!({"protocol_sha256":dataset::hash(PROTOCOL.as_bytes()),"cases":forecasts,"new_model_evaluations":0}),
    )
}
pub fn construct(data: &dataset::dataset::Dataset) -> Result<Prepared> {
    let input = adapter::input(data.task.vocab, &data.task.splits["train"])?;
    let settings = fit::settings()?;
    let (complete_forecast, mut error) = match preflight(&input, &settings, data) {
        Ok(f) => (f, None),
        Err(e) => (Value::Null, Some(e.to_string())),
    };
    let training = if error.is_none() {
        match fit::construct(input, settings) {
            Ok(t) => Some(t),
            Err(e) => {
                error = Some(e.to_string());
                None
            }
        }
    } else {
        None
    };
    Ok(Prepared {
        metadata: metadata(data)?,
        training,
        complete_forecast,
        error,
    })
}
pub fn stage(root: &Path, case: &str) -> Result<()> {
    let inv = inventory(root)?;
    let data = rebuild_data(&inv, case)?;
    let prepared = construct(&data)?;
    let dir = root.join(case);
    fs::create_dir(&dir)?;
    write(&dir.join("state.json"), &prepared)?;
    write(
        &dir.join("seal.json"),
        &json!({"case":case,"state_sha256":digest(&dir.join("state.json"))?,
        "metadata_sha256":artifacts::canonical_digest(&prepared.metadata)?,"endpoint_count":6,
        "all_modes_and_priors_sealed_before_scoring":true}),
    )
}
fn all_sealed(root: &Path) -> Result<()> {
    for case in cases() {
        let dir = root.join(&case);
        let seal: Value = read(&dir.join("seal.json"))?;
        if seal["case"] != case
            || seal["endpoint_count"] != 6
            || seal["all_modes_and_priors_sealed_before_scoring"] != true
            || seal["state_sha256"] != digest(&dir.join("state.json"))?
        {
            return Err(
                "all thirteen cases and six endpoints must be sealed before scoring".into(),
            );
        }
    }
    Ok(())
}
fn load(
    root: &Path,
    inv: &Value,
    case: &str,
) -> Result<(dataset::dataset::Dataset, Prepared, Value)> {
    let data = rebuild_data(inv, case)?;
    let dir = root.join(case);
    let saved: Prepared = read(&dir.join("state.json"))?;
    let seal: Value = read(&dir.join("seal.json"))?;
    if seal["state_sha256"] != digest(&dir.join("state.json"))? {
        return Err("training state seal changed".into());
    }
    let rebuilt = construct(&data)?;
    if saved != rebuilt {
        return Err("complete discovery, policies or six joint CE trajectories differ".into());
    }
    Ok((data, rebuilt, seal))
}

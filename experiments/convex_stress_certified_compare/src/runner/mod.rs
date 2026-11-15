//! Immutable producer/input binding and complete training seals before scores.
//! Adapted from task347 at1371ae4f. Preserve complete344/346/347 histories and
//! original35, then independently repeat the same thirteen-case cohort.
use crate::{
    artifacts::{digest, read, write},
    *,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{fs, path::Path, process::Command};
mod score;
mod summary;
mod verify;
pub use score::{replay, score};
pub use summary::summary;
pub use verify::verify_pair;
#[cfg(test)]
mod binding_tests;
#[cfg(test)]
#[path = "../../../convex_stress_streaming_counts/src/runner/tests.rs"]
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
fn references() -> [(&'static str, &'static str, &'static str); 10] {
    [
        (
            "baseline344.json",
            "baseline344.json",
            "baseline344_report_sha256",
        ),
        (
            "baseline346.json",
            "baseline346.json",
            "baseline346_report_sha256",
        ),
        (
            "basis-manifest.json",
            "basis-manifest.json",
            "baseline346_manifest_sha256",
        ),
        (
            "basis-verified.json",
            "basis-verified.json",
            "baseline346_verified_sha256",
        ),
        (
            "baseline347.json",
            "report.json",
            "baseline347_report_sha256",
        ),
        (
            "stream-manifest.json",
            "manifest.json",
            "baseline347_manifest_sha256",
        ),
        ("stream-verified.json", "", "baseline347_verified_sha256"),
        ("history.json", "history.json", "history_sha256"),
        ("data345.json", "data345.json", "data345_sha256"),
        (
            "original35.json",
            "original35.json",
            "original35_summary_sha256",
        ),
    ]
}
fn checked_reference(path: &Path, expected: &Value) -> Result<()> {
    if fs::metadata(path)?.len() > 1024u64.pow(3)
        || digest(path)? != expected.as_str().ok_or("missing reference hash")?
    {
        return Err("complete inherited reference size/hash differs".into());
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Prepared {
    pub metadata: Value,
    pub training: Option<fit::Stage>,
    pub complete_forecast: Value,
    pub error: Option<String>,
}

fn original_state_digest(prepared: &Prepared, original_forecast: &Value) -> Result<String> {
    // Preserve original Prepared's declaration/serialization order, including
    // every discovery record and optimizer trajectory, without cloning them.
    #[derive(Serialize)]
    struct Original<'a> {
        metadata: &'a Value,
        training: &'a Option<fit::Stage>,
        complete_forecast: &'a Value,
        error: &'a Option<String>,
    }
    artifacts::canonical_digest(&Original {
        metadata: &prepared.metadata,
        training: &prepared.training,
        complete_forecast: original_forecast,
        error: &prepared.error,
    })
}

fn unchanged_training(root: &Path, case: &str, prepared: &Prepared) -> Result<String> {
    let original: Value = read(&root.join("baseline347.json"))?;
    let matching: Vec<_> = original["cases"]
        .as_array()
        .ok_or("missing original347 cases")?
        .iter()
        .filter(|r| r["results"]["case"] == case)
        .collect();
    if matching.len() != 1 {
        return Err("original347 case identity differs".into());
    }
    let row = matching[0];
    let digest = original_state_digest(prepared, &row["results"]["complete_case_forecast"])?;
    if row["seal"]["state_sha256"] != digest {
        return Err("training or discovery changed from complete original347 state".into());
    }
    Ok(digest)
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

pub fn prepare(root: &Path, producer: &str, basis: &Path, verified: &Path) -> Result<()> {
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
    let mut largest = 0u64;
    let mut reference_bytes = 0u128;
    for (_, input, key) in references() {
        let source = if input.is_empty() {
            verified.to_path_buf()
        } else {
            basis.join(input)
        };
        checked_reference(&source, &p[key])?;
        let size = fs::metadata(source)?.len();
        largest = largest.max(size);
        reference_bytes += size as u128;
    }
    let buffer = 64 * largest as u128 + 4 * 1024u128.pow(3);
    if buffer > joint::MAX_BYTES {
        return Err("complete reference parse/copy forecast refused".into());
    }
    let verification: Value = read(verified)?;
    if verification["independently_reproduced"] != true
        || verification["catalog_union_all_modes_all_heldout99_cases"] != 77
        || verification["report_sha256"] != p["baseline347_report_sha256"]
        || verification["manifest_sha256"] != p["baseline347_manifest_sha256"]
    {
        return Err("complete independently verified347 evidence differs".into());
    }
    let source: Value = read(&basis.join("baseline344.json"))?;
    if source["complete"] != true
        || source["new_cases"] != 13
        || source["catalog_all_modes_all_heldout99_cases"] != 69
        || source["new_all_modes_all_heldout99_cases"] != 11
        || source["inventory"]["data_cases"] != source["baseline343"]["reference340"]["cases"]
        || artifacts::canonical_digest(&source["baseline343"])? != p["baseline343_report_sha256"]
        || artifacts::canonical_digest(&source["baseline343"]["reference340"])?
            != p["data_inventory_sha256"]
    {
        return Err("incomplete344 history or original340 inventory".into());
    }
    let data_cases = source["inventory"]["data_cases"].clone();
    drop(source);
    let mut inv = json!({"producer_commit":producer,"binary_sha256":binary()?,"protocol":p,
        "protocol_sha256":dataset::hash(PROTOCOL.as_bytes()),"cases":cases(),"data_cases":data_cases,
        "reference_bytes":reference_bytes as u64,"reference_buffer_bytes":buffer as u64});
    let mut disk = 2 * reference_bytes;
    for case in cases() {
        let data = rebuild_data(&inv, &case)?;
        let bytes = disk::endpoint(
            data.task.vocab,
            fit::settings()?.max_programs,
            &data.task.splits,
        )?;
        // Original/replay raw vectors in each independent run, plus state slack.
        disk += 24 * bytes as u128 + 2 * 1024u128.pow(3);
    }
    if disk > 256 * 1024u128.pow(3)
        || disk > available_disk(root.parent().ok_or("missing storage parent")?)?
    {
        return Err("complete independent-pair disk forecast refused".into());
    }
    inv["complete_pair_disk_bytes"] = json!(disk as u64);
    fs::create_dir(root)?;
    for (file, input, _) in references() {
        fs::copy(
            if input.is_empty() {
                verified.to_path_buf()
            } else {
                basis.join(input)
            },
            root.join(file),
        )?;
    }
    write(&root.join("inventory.json"), &inv)
}

fn available_disk(path: &Path) -> Result<u128> {
    let out = Command::new("df")
        .arg("--output=avail")
        .arg("-B1")
        .arg(path)
        .output()?;
    if !out.status.success() {
        return Err("native free-disk query failed".into());
    }
    Ok(String::from_utf8(out.stdout)?
        .split_whitespace()
        .last()
        .ok_or("missing disk size")?
        .parse()?)
}

fn inventory(root: &Path) -> Result<Value> {
    let inv: Value = read(&root.join("inventory.json"))?;
    let p = protocol()?;
    if inv["binary_sha256"] != binary()?
        || inv["protocol"] != p
        || inv["protocol_sha256"] != dataset::hash(PROTOCOL.as_bytes())
        || inv["cases"] != json!(cases())
    {
        return Err("binary, protocol or complete references changed".into());
    }
    for (file, _, key) in references() {
        checked_reference(&root.join(file), &p[key])?;
    }
    Ok(inv)
}
pub fn repeat_run(source: &Path, root: &Path) -> Result<()> {
    if root.exists() {
        return Err("fresh repeat directory required".into());
    }
    inventory(source)?;
    fs::create_dir(root)?;
    for file in std::iter::once("inventory.json").chain(references().into_iter().map(|r| r.0)) {
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
    let work = 3 * train.scalar_work as u128 + 24 * scoring.scalar_work as u128;
    let buffer = train.buffer_bytes as u128 + scoring.buffer_bytes as u128 + 2 * 1024u128.pow(3);
    let disk =
        12 * disk::endpoint(input.vocabulary, settings.max_programs, &data.task.splits)? as u128;
    let complete_forecast = json!({"one_construction":train,"one_endpoint_evaluation":scoring,
            "construction_passes":3,"scoring_passes_per_endpoint":4,"endpoints":6,
            "complete_scalar_work":work as u64,"complete_buffer_bytes":buffer as u64,"raw_disk_bytes":disk as u64,
            "process_address_space_bytes":68719476736u64,
            "scope":"all candidate search, six training traces, two scored product audits plus two raw readbacks and fresh reconstruction; rational products counted separately from their nonconstant limb cost; hashing, JSON and data regeneration measured additionally"});
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
    let original_hash = unchanged_training(root, case, &prepared)?;
    let dir = root.join(case);
    fs::create_dir(&dir)?;
    write(&dir.join("state.json"), &prepared)?;
    write(
        &dir.join("seal.json"),
        &json!({"case":case,"state_sha256":digest(&dir.join("state.json"))?,
        "metadata_sha256":artifacts::canonical_digest(&prepared.metadata)?,"endpoint_count":6,
        "original347_state_sha256":original_hash,
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
    if seal["original347_state_sha256"] != unchanged_training(root, case, &rebuilt)? {
        return Err("original347 training binding changed".into());
    }
    Ok((data, rebuilt, seal))
}

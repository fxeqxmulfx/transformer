//! Freeze/bind all sources and original data, then seal all eight trained
//! cases before any heldout measurement. Independent repeat rebuilds all data
//! and all trained states; complete earlier histories remain byte-identical.
use crate::*;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{fs, path::Path, process::Command};
mod score;
mod summary;
pub use score::score;
pub use summary::{summary, verify_pair};

fn protocol() -> Result<Value> {
    Ok(serde_json::from_str(PROTOCOL)?)
}
fn binary() -> Result<String> {
    artifacts::digest(&std::env::current_exe()?)
}
fn cases() -> Vec<String> {
    basis::CASES.iter().map(|s| s.to_string()).collect()
}
fn references() -> [(&'static str, &'static str); 4] {
    [
        ("data345.json", "data_inventory_sha256"),
        ("baseline344.json", "baseline344_sha256"),
        ("original35.json", "original35_sha256"),
        ("history.json", "history_sha256"),
    ]
}
fn checked_reference(path: &Path, expected: &Value) -> Result<()> {
    if fs::metadata(path)?.len() > 1024u64.pow(3)
        || artifacts::digest(path)? != expected.as_str().ok_or("missing reference digest")?
    {
        return Err("complete reference size or digest differs".into());
    }
    Ok(())
}
fn data_metadata(inv: &Value, case: &str) -> Result<Value> {
    let matches: Vec<_> = inv["data_cases"]
        .as_array()
        .ok_or("missing eight data cases")?
        .iter()
        .filter(|c| c["case"] == case)
        .collect();
    if matches.len() != 1 {
        return Err("missing or duplicated original case".into());
    }
    Ok(matches[0]["data"].clone())
}
fn rebuild(inv: &Value, case: &str) -> Result<basis::dataset::Dataset> {
    let (data, noise) = from_data(basis::dataset::build(case))?;
    if from_data(basis::dataset::metadata(&data, &noise))? != data_metadata(inv, case)? {
        return Err("complete original dataset metadata differs".into());
    }
    Ok(data)
}
fn inventory(root: &Path) -> Result<Value> {
    let inv: Value = artifacts::read(&root.join("inventory.json"))?;
    let p = protocol()?;
    if inv["binary_sha256"] != binary()?
        || inv["protocol"] != p
        || inv["protocol_sha256"] != basis::hash(PROTOCOL.as_bytes())
        || inv["cases"] != json!(cases())
    {
        return Err("frozen executable, protocol or complete case list differs".into());
    }
    for (file, key) in references() {
        checked_reference(&root.join(file), &p[key])?;
    }
    Ok(inv)
}
fn available_disk(path: &Path) -> Result<u128> {
    let output = Command::new("df")
        .arg("--output=avail")
        .arg("-B1")
        .arg(path)
        .output()?;
    if !output.status.success() {
        return Err("native available-disk query failed".into());
    }
    Ok(String::from_utf8(output.stdout)?
        .split_whitespace()
        .last()
        .ok_or("missing disk size")?
        .parse()?)
}

pub fn prepare(data_root: &Path, root: &Path, producer: &str) -> Result<()> {
    if root.exists() || producer.len() != 40 || !producer.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("fresh run root and full committed producer required".into());
    }
    let head = Command::new("git").args(["rev-parse", "HEAD"]).output()?;
    let status = Command::new("git")
        .args(["status", "--porcelain"])
        .output()?;
    if !head.status.success()
        || !status.status.success()
        || !status.stdout.is_empty()
        || String::from_utf8(head.stdout)?.trim() != producer
    {
        return Err("clean complete producer must be committed before measurement".into());
    }
    let p = protocol()?;
    for (file, key) in references() {
        checked_reference(
            &data_root.join(if file == "data345.json" {
                "inventory.json"
            } else {
                file
            }),
            &p[key],
        )?;
    }
    let data: Value = artifacts::read(&data_root.join("inventory.json"))?;
    if data["new_data_cases"] != 8
        || data["new_model_evaluations"] != 0
        || data["new_accuracy_claims"] != 0
    {
        return Err("unreviewed or incomplete eight-case data inventory".into());
    }
    let mut inv = json!({"producer_commit":producer,"binary_sha256":binary()?,"protocol":p,
        "protocol_sha256":basis::hash(PROTOCOL.as_bytes()),"cases":cases(),"data_cases":data["cases"],
        "process_address_space_bytes":68719476736u64});
    let mut forecasts = vec![];
    let mut disk = 0u128;
    for case in cases() {
        let dataset = rebuild(&inv, &case)?;
        let f = evaluation::forecast(dataset.task.vocab, 1, &dataset.task.splits)?;
        // Six raw endpoints plus a generous half-GiB serialized-state bound,
        // metadata and traces, in each of the two independent runs.
        disk += 12 * f.disk_bytes as u128 + 2 * 1024u128.pow(3);
        forecasts.push(json!({"case":case,"one_endpoint":f}));
    }
    for (file, _) in references() {
        disk += 2 * fs::metadata(data_root.join(if file == "data345.json" {
            "inventory.json"
        } else {
            file
        }))?
        .len() as u128;
    }
    if disk > 256 * 1024u128.pow(3)
        || disk > available_disk(root.parent().ok_or("missing storage parent")?)?
    {
        return Err("complete original/repeat disk forecast refused".into());
    }
    inv["scoring_forecasts"] = json!(forecasts);
    inv["complete_pair_disk_bytes"] = json!(disk as u64);
    fs::create_dir(root)?;
    for (file, _) in references() {
        fs::copy(
            data_root.join(if file == "data345.json" {
                "inventory.json"
            } else {
                file
            }),
            root.join(file),
        )?;
    }
    artifacts::write(&root.join("inventory.json"), &inv)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Prepared {
    pub metadata: Value,
    pub stage: Option<fit::Stage>,
    pub error: Option<String>,
}

fn read_prepared(root: &Path, case: &str) -> Result<Prepared> {
    let path = root.join(format!("{case}.stage.json"));
    if 64 * fs::metadata(&path)?.len() as u128 + 12 * 1024u128.pow(3) > joint::MAX_BYTES {
        return Err("prepared-state plus data parse forecast refused".into());
    }
    artifacts::read(&path)
}
pub fn train(root: &Path) -> Result<()> {
    let inv = inventory(root)?;
    if root.join("train.seal.json").exists() {
        return Err("all-case training seal already exists".into());
    }
    let mut records = vec![];
    for case in cases() {
        let dataset = rebuild(&inv, &case)?;
        let forecast = evaluation::forecast(dataset.task.vocab, 1, &dataset.task.splits)?;
        let result = fit::construct_bounded(
            dataset.task.vocab,
            &dataset.task.splits["train"],
            Some(forecast),
        );
        let (stage, error) = match result {
            Ok(stage) => (Some(stage), None),
            Err(e) => (None, Some(e.to_string())),
        };
        let prepared = Prepared {
            metadata: data_metadata(&inv, &case)?,
            stage,
            error,
        };
        let file = format!("{case}.stage.json");
        artifacts::write(&root.join(&file), &prepared)?;
        let fitted = prepared.stage.as_ref().map_or(0, |s| {
            s.endpoints.iter().filter(|e| e.error.is_none()).count()
        });
        println!("{case}: fitted {fitted}/6, refusal={:?}", prepared.error);
        records
            .push(json!({"case":case,"file":file,"sha256":artifacts::digest(&root.join(&file))?}));
    }
    artifacts::write(
        &root.join("train.seal.json"),
        &json!({"inventory_sha256":artifacts::digest(&root.join("inventory.json"))?,
        "all_eight_cases_precede_any_heldout_score":true,"cases":records}),
    )
}
fn seal(root: &Path) -> Result<Value> {
    let saved: Value = artifacts::read(&root.join("train.seal.json"))?;
    if saved["inventory_sha256"] != artifacts::digest(&root.join("inventory.json"))?
        || saved["all_eight_cases_precede_any_heldout_score"] != true
    {
        return Err("all-case training seal changed".into());
    }
    let records = saved["cases"].as_array().ok_or("missing sealed cases")?;
    if records.len() != 8 {
        return Err("incomplete training seal".into());
    }
    for (case, record) in cases().iter().zip(records) {
        let file = format!("{case}.stage.json");
        if record["case"] != *case
            || record["file"] != file
            || record["sha256"] != artifacts::digest(&root.join(&file))?
        {
            return Err("sealed original training stage changed".into());
        }
    }
    Ok(saved)
}
pub fn repeat(original: &Path, root: &Path) -> Result<()> {
    if root.exists() {
        return Err("fresh independent repeat directory required".into());
    }
    let inv = inventory(original)?;
    seal(original)?;
    if inv["complete_pair_disk_bytes"]
        .as_u64()
        .ok_or("missing disk forecast")? as u128
        / 2
        > available_disk(root.parent().ok_or("missing repeat parent")?)?
    {
        return Err("independent repeat disk forecast refused".into());
    }
    fs::create_dir(root)?;
    for file in [
        "inventory.json",
        "data345.json",
        "baseline344.json",
        "original35.json",
        "history.json",
    ] {
        fs::copy(original.join(file), root.join(file))?;
    }
    train(root)?;
    if artifacts::digest(&root.join("train.seal.json"))?
        != artifacts::digest(&original.join("train.seal.json"))?
    {
        return Err("independent complete training stages differ".into());
    }
    score(root)?;
    summary(root)
}

//! Validate original historical criteria and save every source-derived split.
//! Complete344 and original35 reports are preserved as exact separate files.
use crate::*;
pub use sequences::runner::artifacts::digest;
use serde_json::{Value, json};
use std::{
    fs,
    io::{BufWriter, Write},
    path::Path,
    process::Command,
};

fn write(path: &Path, value: &Value) -> Result<()> {
    let file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    let mut writer = BufWriter::with_capacity(1024 * 1024, file);
    serde_json::to_writer(&mut writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    Ok(())
}
fn checked_history(path: &Path) -> Result<Value> {
    let p: Value = serde_json::from_str(PROTOCOL)?;
    if fs::metadata(path)?.len() > 16 * 1024 * 1024 || digest(path)? != p["history_sha256"] {
        return Err("source history size or hash differs".into());
    }
    Ok(serde_json::from_reader(fs::File::open(path)?)?)
}
pub fn cases(history: &Value) -> Result<Vec<Value>> {
    let mut out = vec![];
    for case in CASES {
        let (d, noise) = dataset::build(case)?;
        let matches: Vec<_> = history["runs"]
            .as_array()
            .ok_or("missing historical cases")?
            .iter()
            .filter(|r| r["case"] == case)
            .collect();
        if matches.len() != 1 {
            return Err("missing or duplicate Basis history".into());
        }
        let old = matches[0];
        if old["metric"] != "sequence_accuracy" || old["target"] != 0.99 || old["seed"] != 1 {
            return Err("source Basis criterion changed".into());
        }
        let metadata = dataset::metadata(&d, &noise)?;
        let mut tests = 0;
        if let Some(scores) = old["final_test"].as_object() {
            for (name, m) in scores {
                let counts = &metadata["records"][name];
                if m["target_count"] != counts["target_count"]
                    || m["example_count"] != counts["examples"]
                    || m["transition_count"] != counts["transition_count"]
                {
                    return Err(format!("historical denominator changed for{case}/{name}").into());
                }
                tests += 1;
            }
            if tests
                != d.task
                    .splits
                    .keys()
                    .filter(|name| name.as_str() == "test" || name.starts_with("test/"))
                    .count()
            {
                return Err("historical test omitted".into());
            }
        } else if !old["final_test"].is_null() || old["status"] != "pending" {
            return Err("missing historical measurements not marked pending".into());
        }
        out.push(json!({"case":case,"data":metadata,"historical_gpt":old,
            "checked_historical_test_splits":tests,"historical_denominators_available":tests>0,
            "new_model_results":null}));
    }
    Ok(out)
}
pub fn check_history(history: &Path, output: &Path) -> Result<()> {
    let history = checked_history(history)?;
    write(
        output,
        &json!({"cases":cases(&history)?,"new_model_evaluations":0}),
    )
}
fn source_binding(producer: &str) -> Result<()> {
    let head = Command::new("git").args(["rev-parse", "HEAD"]).output()?;
    let status = Command::new("git")
        .args(["status", "--porcelain"])
        .output()?;
    if producer.len() != 40
        || !producer.bytes().all(|b| b.is_ascii_hexdigit())
        || !head.status.success()
        || !status.status.success()
        || String::from_utf8(head.stdout)?.trim() != producer
        || !status.stdout.is_empty()
    {
        return Err("clean committed producer required".into());
    }
    Ok(())
}
fn retain(source: &Path, dest: &Path, expected: &Value) -> Result<()> {
    if fs::metadata(source)?.len() > 1 << 30 || digest(source)? != *expected {
        return Err("retained complete report size or hash differs".into());
    }
    fs::copy(source, dest)?;
    if digest(dest)? != *expected {
        return Err("retained report copy differs".into());
    }
    Ok(())
}
pub fn prepare(history: &Path, baseline: &Path, root: &Path, producer: &str) -> Result<()> {
    if root.exists() {
        return Err("fresh output root required".into());
    }
    source_binding(producer)?;
    let source_history = checked_history(history)?;
    let p: Value = serde_json::from_str(PROTOCOL)?;
    let records = cases(&source_history)?;
    fs::create_dir(root)?;
    retain(history, &root.join("history.json"), &p["history_sha256"])?;
    retain(
        &baseline.join("report.json"),
        &root.join("baseline344.json"),
        &p["baseline344_sha256"],
    )?;
    retain(
        &baseline.join("original35.json"),
        &root.join("original35.json"),
        &p["original35_sha256"],
    )?;
    write(
        &root.join("inventory.json"),
        &json!({"producer_commit":producer,
        "binary_sha256":digest(&std::env::current_exe()?)?,"protocol":p,"protocol_sha256":hash(PROTOCOL.as_bytes()),
        "cases":records,"new_data_cases":8,"new_model_evaluations":0,"new_accuracy_claims":0,
        "baseline344_reference":{"file":"baseline344.json","sha256":p["baseline344_sha256"],"whole_report_preserved":true},
        "original35_reference":{"file":"original35.json","sha256":p["original35_sha256"],"new_evaluations":0},
        "history_reference":{"file":"history.json","sha256":p["history_sha256"]},
        "buffer_forecast_bytes":8u64*1024*1024*1024,
        "process_address_space_bytes":68719476736u64}),
    )
}

/// Rebuild every pool independently from the full pinned historical input.
/// Saved generated splits and partial historical fields are never reused.
pub fn repeat(original: &Path, root: &Path) -> Result<()> {
    if root.exists() {
        return Err("fresh repeat root required".into());
    }
    let saved: Value = serde_json::from_reader(fs::File::open(original.join("inventory.json"))?)?;
    let p: Value = serde_json::from_str(PROTOCOL)?;
    if saved["protocol"] != p
        || saved["protocol_sha256"] != hash(PROTOCOL.as_bytes())
        || saved["binary_sha256"] != digest(&std::env::current_exe()?)?
    {
        return Err("independent producer binding changed".into());
    }
    let historical = checked_history(&original.join("history.json"))?;
    if serde_json::to_value(cases(&historical)?)? != saved["cases"] {
        return Err("independent full data reconstruction differs".into());
    }
    fs::create_dir(root)?;
    retain(
        &original.join("history.json"),
        &root.join("history.json"),
        &p["history_sha256"],
    )?;
    retain(
        &original.join("baseline344.json"),
        &root.join("baseline344.json"),
        &p["baseline344_sha256"],
    )?;
    retain(
        &original.join("original35.json"),
        &root.join("original35.json"),
        &p["original35_sha256"],
    )?;
    write(&root.join("inventory.json"), &saved)
}

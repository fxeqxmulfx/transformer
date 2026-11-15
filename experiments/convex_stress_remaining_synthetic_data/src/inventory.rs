//! Authenticate all source histories and preserve the complete reviewed339
//! report plus a separately copied original35 reference. No model is trained.
use crate::*;
pub use generative::inventory::read_pinned;
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
fn checked_cases(history: &Value) -> Result<Vec<Value>> {
    let mut cases = vec![];
    for case in CASES {
        let d = dataset::build(case)?;
        let old: Vec<_> = history["runs"]
            .as_array()
            .ok_or("missing history")?
            .iter()
            .filter(|r| r["case"] == case)
            .collect();
        if old.len() != 1 {
            return Err("missing or duplicate case history".into());
        }
        let mut tests = 0;
        for (name, m) in old[0]["final_test"]
            .as_object()
            .ok_or("missing original tests")?
        {
            let rows = d.task.splits.get(name).ok_or("historical split missing")?;
            let targets: usize = rows
                .iter()
                .map(|e| e.targets.iter().filter(|&&y| y >= 0).count())
                .sum();
            let changes: usize = rows
                .iter()
                .map(|e| e.changes.iter().filter(|&&b| b).count())
                .sum();
            if m["target_count"] != targets
                || m["example_count"] != rows.len()
                || m["transition_count"] != changes
            {
                return Err(format!("historical denominator differs for {case}/{name}: targets={targets},transitions={changes}").into());
            }
            tests += 1;
        }
        if tests
            != d.task
                .splits
                .keys()
                .filter(|n| n.as_str() == "test" || n.starts_with("test/"))
                .count()
        {
            return Err("original test omitted from history check".into());
        }
        cases.push(json!({"case":case,"data":dataset::metadata(&d)?,"historical_gpt":old[0],
            "historical_test_denominators_match":true,"checked_test_splits":tests,"new_model_results":null}));
    }
    Ok(cases)
}
pub fn check_history(history: &Path, output: &Path) -> Result<()> {
    let p: Value = serde_json::from_str(PROTOCOL)?;
    let h = read_pinned(
        history,
        p["history_sha256"].as_str().ok_or("missing history pin")?,
    )?;
    write(
        output,
        &json!({"cases":checked_cases(&h)?,"new_model_evaluations":0}),
    )
}
pub fn prepare(history: &Path, baseline_root: &Path, root: &Path, producer: &str) -> Result<()> {
    if root.exists() || producer.len() != 40 || !producer.bytes().all(|b| b.is_ascii_hexdigit()) {
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
    let p: Value = serde_json::from_str(PROTOCOL)?;
    let baseline = baseline_root.join("report.json");
    let original = baseline_root.join("original35.json");
    let baseline_bytes = fs::metadata(&baseline)?.len();
    let history_bytes = fs::metadata(history)?.len();
    let original_bytes = fs::metadata(&original)?.len();
    let buffer = 64u128 * (baseline_bytes as u128 + history_bytes as u128)
        + (4u128 << 30)
        + original_bytes as u128;
    if baseline_bytes > 1 << 30
        || history_bytes > 1 << 30
        || original_bytes > 1 << 30
        || buffer > 56 * 1024u128.pow(3)
    {
        return Err("complete parse/copy buffer forecast refused".into());
    }
    let digest = sequences::runner::artifacts::digest;
    if digest(&original)? != p["original35_sha256"] {
        return Err("original35 reference changed".into());
    }
    let baseline = read_pinned(
        &baseline,
        p["baseline339_sha256"]
            .as_str()
            .ok_or("missing baseline pin")?,
    )?;
    if baseline["complete"] != true
        || baseline["cohort_cases"] != 23
        || baseline["fully_measured_new_cases"] != 7
        || baseline["original35_reference"]["sha256"] != p["original35_sha256"]
    {
        return Err("reviewed complete339 report required".into());
    }
    let history = read_pinned(
        history,
        p["history_sha256"].as_str().ok_or("missing history pin")?,
    )?;
    let cases = checked_cases(&history)?;
    fs::create_dir(root)?;
    fs::copy(original, root.join("original35.json"))?;
    write(
        &root.join("inventory.json"),
        &json!({"producer_commit":producer,"binary_sha256":digest(&std::env::current_exe()?)?,
        "protocol":p,"protocol_sha256":hash(PROTOCOL.as_bytes()),"cases":cases,"baseline339":baseline,
        "original35_reference":{"file":"original35.json","sha256":p["original35_sha256"]},
        "new_model_evaluations":0,"new_data_cases":13,"prepared_additional_cases":36,
        "additional_cases_previously_scored":23,"additional_cases_still_unmeasured":107,
        "all165_evaluated":false,"historical_case_fingerprints_available":false,
        "buffer_forecast_bytes":buffer as u64,"baseline_bytes":baseline_bytes,
        "history_bytes":history_bytes,"original35_bytes":original_bytes}),
    )
}

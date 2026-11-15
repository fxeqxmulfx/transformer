//! Read-only post-run audit. This binary is separate from the frozen producer;
//! it never fits a model, selects a program, modifies artifacts or scores labels.
use convex_stress_synthetic_programs::{Result, artifacts, dataset, runner};
use serde_json::{Value, json};
use std::path::Path;

fn audit(root: &Path, binary: &Path, producer: &str) -> Result<Value> {
    let p = runner::protocol()?;
    let inv: Value = artifacts::read(&root.join("inventory.json"))?;
    if inv["producer_commit"] != producer
        || inv["binary_sha256"] != artifacts::digest(binary)?
        || inv["protocol"] != p
        || inv["protocol_sha256"]
            != dataset::hash(convex_stress_synthetic_programs::PROTOCOL.as_bytes())
    {
        return Err("committed producer/binary/protocol binding differs".into());
    }
    let report: Value = artifacts::read(&root.join("report.json"))?;
    if report["inventory"] != inv
        || report["complete"] != true
        || report["cohort_cases"] != 36
        || report["new_cases"] != 13
        || report["inherited_cases"] != 23
    {
        return Err("saved report omits the complete cohort or inventory".into());
    }
    let reference_hash = artifacts::canonical_digest(&report["reference340"])?;
    let baseline_hash = artifacts::canonical_digest(&report["reference340"]["baseline339"])?;
    let original_hash = artifacts::digest(&root.join("original35.json"))?;
    if reference_hash != p["data_inventory_sha256"]
        || artifacts::digest(&root.join("reference-data.json"))? != reference_hash
        || baseline_hash != p["baseline339_report_sha256"]
        || report["baseline339_exact_sha256"] != baseline_hash
        || original_hash != p["original35_summary_sha256"]
        || report["original35_reference"]["sha256"] != original_hash
        || report["original35_reference"]["new_evaluations"] != 0
    {
        return Err("whole340/339/original35 history or separate reference changed".into());
    }
    let saved_cases = report["cases"]
        .as_array()
        .ok_or("missing saved case records")?;
    if saved_cases.len() != 13 {
        return Err("thirteen-case denominator changed".into());
    }
    let (mut measured, mut successful, mut raw_files) = (0usize, 0usize, 0usize);
    for case in dataset::CASES {
        let dir = root.join(case);
        let metric: Value = artifacts::read(&dir.join("metrics.json"))?;
        let replay: Value = artifacts::read(&dir.join("replay.json"))?;
        let seal: Value = artifacts::read(&dir.join("seal.json"))?;
        let matches: Vec<_> = saved_cases
            .iter()
            .filter(|v| v["results"]["case"] == case)
            .collect();
        if matches.len() != 1
            || matches[0] != &metric
            || metric["seal"] != seal
            || seal["state_sha256"] != artifacts::digest(&dir.join("state.json"))?
            || replay
                != json!({"case":case,"exact_replay":true,"state_sha256":seal["state_sha256"],
                "metrics_sha256":artifacts::digest(&dir.join("metrics.json"))?})
            || artifacts::digest(&dir.join("replay/metrics.json"))?
                != artifacts::digest(&dir.join("metrics.json"))?
        {
            return Err("case state, metric or fresh-replay binding changed".into());
        }
        let endpoints = metric["results"]["endpoints"]
            .as_array()
            .ok_or("missing endpoints")?;
        if endpoints.len() != 6 {
            return Err("six-endpoint denominator changed".into());
        }
        let coordinates: Vec<_> = endpoints
            .iter()
            .filter(|e| e["prior"] == "Coordinates")
            .collect();
        if coordinates.len() != 3 {
            return Err("coordinate mode denominator differs".into());
        }
        measured += usize::from(
            coordinates
                .iter()
                .all(|e| e["error"].is_null() && e["measurement"].is_object()),
        );
        successful += usize::from(coordinates.iter().all(|e| e["all_heldout99"] == true));
        for (i, e) in endpoints.iter().enumerate() {
            let file = format!("raw-{i}.jsonl");
            if e["measurement"].is_object() {
                if e["measurement"]["raw_file"] != file
                    || e["measurement"]["raw_sha256"] != artifacts::digest(&dir.join(&file))?
                    || artifacts::digest(&dir.join("replay").join(&file))?
                        != e["measurement"]["raw_sha256"]
                {
                    return Err("raw complete products changed after measurement".into());
                }
                raw_files += 2;
                for split in e["measurement"]["splits"]
                    .as_object()
                    .ok_or("missing original splits")?
                    .values()
                {
                    let m = &split["metrics"];
                    let ratio = |a: &str, b: &str| -> Result<f64> {
                        let denominator = m[b].as_u64().ok_or("invalid denominator")?;
                        if denominator == 0 {
                            return Err("zero score denominator".into());
                        }
                        Ok(m[a].as_u64().ok_or("invalid correct count")? as f64
                            / denominator as f64)
                    };
                    let (raw, exact) = (ratio("correct", "targets")?, ratio("exact", "examples")?);
                    if m["token_accuracy"] != json!(raw)
                        || m["sequence_accuracy"] != json!(exact)
                        || split["goals"]
                            != json!({"original_success":exact>=0.95,"raw99":raw>=0.99,"exact99":exact>=0.99,"both99":raw>=0.99 && exact>=0.99})
                    {
                        return Err("stored accuracy or original/raw99/exact99 decision differs from counts".into());
                    }
                }
            } else if e["error"].is_null() || e["all_heldout99"] != false {
                return Err("unmeasured refusal counted as a success".into());
            }
        }
    }
    if report["fully_measured_new_cases"] != measured
        || report["new_all_modes_all_heldout99_cases"] != successful
        || report["measured_catalog_cases"] != 58 + measured
        || report["catalog_all_modes_all_heldout99_cases"] != 58 + successful
        || report["remaining_unmeasured_catalog_cases"] != 107 - measured
    {
        return Err("catalog scope counts differ from complete measured endpoints".into());
    }
    Ok(
        json!({"producer_commit":producer,"whole340_sha256":reference_hash,"whole339_sha256":baseline_hash,
        "separate_original35_sha256":original_hash,"case_records":13,"complete_raw_files_rehashed":raw_files,
        "new_measured_cases":measured,"new_all_modes_all_heldout99_cases":successful,
        "report_sha256":artifacts::digest(&root.join("report.json"))?}),
    )
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let [root, binary, producer] = args.as_slice() else {
        return Err("usage: audit-retained RUN FROZEN_BINARY PRODUCER_COMMIT".into());
    };
    println!(
        "{}",
        serde_json::to_string(&audit(Path::new(root), Path::new(binary), producer)?)?
    );
    Ok(())
}

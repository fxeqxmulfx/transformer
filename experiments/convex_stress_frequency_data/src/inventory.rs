//! Preserve complete reviewed334 results and original GPT frequency histories.
//! This inventory constructs no model and makes no new accuracy claim.
use crate::*;
pub use generative::inventory::read_pinned;
use serde_json::{Value, json};
use std::path::Path;

pub fn prepare(history: &Path, baseline: &Path, output: &Path) -> Result<()> {
    if output.exists() {
        return Err("fresh frequency inventory output required".into());
    }
    let p: Value = serde_json::from_str(PROTOCOL)?;
    let baseline_bytes = std::fs::metadata(baseline)?.len();
    let history_bytes = std::fs::metadata(history)?.len();
    let buffer_bytes = 64u128 * (baseline_bytes as u128 + history_bytes as u128) + (4u128 << 30);
    if baseline_bytes > 1 << 30 || history_bytes > 1 << 30 || buffer_bytes > 56 * 1024u128.pow(3) {
        return Err("complete inventory parse and copy allowance refused".into());
    }
    let history = read_pinned(
        history,
        p["history_sha256"].as_str().ok_or("missing history pin")?,
    )?;
    let baseline = read_pinned(
        baseline,
        p["baseline334_sha256"]
            .as_str()
            .ok_or("missing baseline pin")?,
    )?;
    if baseline["cohort_cases"] != 16 || baseline["complete"] != true {
        return Err("complete reviewed sixteen-case baseline required".into());
    }
    let mut cases = vec![];
    for case in CASES {
        let data = dataset::build(case)?;
        let old: Vec<_> = history["runs"]
            .as_array()
            .ok_or("missing history")?
            .iter()
            .filter(|r| r["case"] == case)
            .collect();
        if old.len() != 1 {
            return Err("missing or duplicated frequency history".into());
        }
        for (split, m) in old[0]["final_test"]
            .as_object()
            .ok_or("missing historical tests")?
        {
            let rows = data
                .task
                .splits
                .get(split)
                .ok_or("historical split absent")?;
            let targets: usize = rows.iter().map(|e| e.answer.len()).sum();
            let transitions: usize = rows
                .iter()
                .map(|e| e.changes.iter().filter(|&&c| c).count())
                .sum();
            if m["target_count"] != targets
                || m["example_count"] != rows.len()
                || m["transition_count"] != transitions
            {
                return Err(format!("historical layout differs for {case}/{split}").into());
            }
        }
        cases.push(
            json!({"case":case,"data":dataset::metadata(&data),"historical_gpt":old[0],
            "historical_test_denominators_match":true,"new_model_results":null}),
        );
    }
    sequences::runner::artifacts::write(
        output,
        &json!({
            "protocol":p,"protocol_sha256":hash(PROTOCOL.as_bytes()),"cases":cases,
            "baseline334":baseline,"new_model_evaluations":0,"historical_case_fingerprints_available":false,
            "additional_cases_previously_scored":16,"additional_cases_still_unmeasured":114,
            "new_data_cases":7,"complete_cohort_data_cases":23,
            "buffer_forecast_bytes":u64::try_from(buffer_bytes)?,"baseline_bytes":baseline_bytes,"history_bytes":history_bytes
        }),
    )
}

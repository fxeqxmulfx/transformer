//! Pinned original history and complete six-case326 reference. No learned
//! model is constructed or evaluated while preparing this data inventory.
use crate::*;
use serde_json::{Value, json};
use std::path::Path;
pub use typed::inventory::read_pinned;

pub fn prepare(history: &Path, baseline: &Path, output: &Path) -> Result<()> {
    if output.exists() {
        return Err("fresh generative inventory output required".into());
    }
    let p: Value = serde_json::from_str(PROTOCOL)?;
    let history = read_pinned(
        history,
        p["history_sha256"].as_str().ok_or("missing history pin")?,
    )?;
    let baseline = read_pinned(
        baseline,
        p["baseline326_sha256"]
            .as_str()
            .ok_or("missing baseline pin")?,
    )?;
    if baseline["cohort_cases"] != 6 || baseline["complete"] != true {
        return Err("complete reviewed six-case baseline required".into());
    }
    let mut cases = vec![];
    for case in CASES {
        let data = dataset::build(case)?;
        let old: Vec<_> = history["runs"]
            .as_array()
            .ok_or("missing historical runs")?
            .iter()
            .filter(|r| r["case"] == case)
            .collect();
        if old.len() != 1 {
            return Err("missing or duplicated generative historical case".into());
        }
        for (split, m) in old[0]["final_test"]
            .as_object()
            .ok_or("missing historical tests")?
        {
            let rows = data
                .task
                .splits
                .get(split)
                .ok_or("historical test absent from native data")?;
            let targets: usize = rows.iter().map(|e| e.answer.len()).sum();
            if m["target_count"] != targets || m["example_count"] != rows.len() {
                return Err(
                    format!("historical aggregate layout differs for {case}/{split}").into(),
                );
            }
        }
        cases.push(
            json!({"case":case,"data":dataset::metadata(&data),"historical_gpt":old[0],
            "historical_test_denominators_match":true,"new_model_results":null}),
        );
    }
    sequences::runner::artifacts::write(
        output,
        &json!({"protocol":p,"protocol_sha256":hash(PROTOCOL.as_bytes()),
        "cases":cases,"baseline326":baseline,"new_model_evaluations":0,"historical_case_fingerprints_available":false}),
    )
}

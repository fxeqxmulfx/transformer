//! Independent inventory readback: canonical nested-reference hashes and
//! regenerated case metadata, without printing large inherited report trees.
use convex_stress_remaining_synthetic_data::{
    CASES, PROTOCOL, Result, dataset, hash, inventory, sequences,
};
use serde_json::{Value, json};
use std::{fs, path::Path};

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let [root, binary, producer, history] = args.as_slice() else {
        return Err(
            "usage: audit-retained INVENTORY_ROOT FROZEN_BINARY PRODUCER_COMMIT FROZEN_HISTORY"
                .into(),
        );
    };
    let root = Path::new(root);
    let path = root.join("inventory.json");
    let bytes = fs::metadata(&path)?.len();
    let history_bytes = fs::metadata(history)?.len();
    let buffer = 64u128 * (bytes as u128 + history_bytes as u128) + (4u128 << 30);
    if bytes > 1 << 30 || history_bytes > 1 << 30 || buffer > 56 * 1024u128.pow(3) {
        return Err("independent readback buffer forecast refused".into());
    }
    let p: Value = serde_json::from_str(PROTOCOL)?;
    let inv: Value = sequences::runner::artifacts::read(&path)?;
    let digest = sequences::runner::artifacts::digest;
    if inv["protocol"] != p
        || inv["protocol_sha256"] != hash(PROTOCOL.as_bytes())
        || inv["producer_commit"] != *producer
        || inv["binary_sha256"] != digest(Path::new(binary))?
        || inv["new_model_evaluations"] != 0
        || inv["new_data_cases"] != 13
        || inv["additional_cases_previously_scored"] != 23
        || inv["additional_cases_still_unmeasured"] != 107
    {
        return Err("inventory provenance or measurement scope differs".into());
    }
    let mut baseline = serde_json::to_vec(&inv["baseline339"])?;
    baseline.push(b'\n');
    let baseline_hash = hash(&baseline);
    drop(baseline);
    let original_hash = digest(&root.join("original35.json"))?;
    if baseline_hash != p["baseline339_sha256"] || original_hash != p["original35_sha256"] {
        return Err("complete inherited report or original35 copy changed".into());
    }
    let h = inventory::read_pinned(
        Path::new(history),
        p["history_sha256"].as_str().ok_or("missing history pin")?,
    )?;
    let rows = inv["cases"].as_array().ok_or("missing case records")?;
    if rows.len() != 13 {
        return Err("case denominator changed".into());
    }
    for (case, row) in CASES.iter().zip(rows) {
        let matching: Vec<_> = h["runs"]
            .as_array()
            .ok_or("missing history runs")?
            .iter()
            .filter(|r| r["case"] == *case)
            .collect();
        if matching.len() != 1
            || row["case"] != *case
            || row["historical_gpt"] != *matching[0]
            || row["data"] != dataset::metadata(&dataset::build(case)?)?
            || !row["new_model_results"].is_null()
        {
            return Err(format!("full metadata/history changed for {case}").into());
        }
    }
    println!(
        "{}",
        json!({"cases":13,"baseline339_sha256":baseline_hash,"original35_sha256":original_hash,
        "case_metadata_and_full_histories_match":true,"new_model_evaluations":0,
        "buffer_forecast_bytes":buffer as u64})
    );
    Ok(())
}

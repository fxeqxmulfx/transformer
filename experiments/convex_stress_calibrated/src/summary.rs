//! Strict complete-inventory summary, retaining resource and replay failures.
use crate::cohort;
use convex_stress_calibrated::Result;
use serde_json::{Value, json};
use std::{collections::BTreeSet, fs, path::Path};
pub fn summarize(root: &Path) -> Result<()> {
    if fs::read_to_string(root.join("fingerprint.txt"))?.trim() != cohort::fingerprint() {
        return Err("cohort fingerprint mismatch".into());
    }
    let mut records = vec![];
    let mut expected = BTreeSet::new();
    for row in cohort::rows()? {
        for arm in ["scaled", "calibrated"] {
            let name = row["name"].as_str().unwrap();
            let id = format!("{arm}--{name}");
            expected.insert(id.clone());
            let code = fs::read_to_string(root.join(format!("{id}.exit")))?;
            if code.trim() != "0" {
                records.push(json!({"task":name,"tier":row["tier"],"arm":arm,
                    "status":"execution_failure","exit_code":code.trim(),
                    "primary_pass":false,"all_split_pass":false}));
                continue;
            }
            let replay = fs::read_to_string(root.join(format!("{id}.replay.exit")))?;
            if replay.trim() != "0" {
                return Err(format!("failed replay: {id}").into());
            }
            let r: Value = serde_json::from_slice(&fs::read(root.join(&id).join("result.json"))?)?;
            if r["source_fingerprint"] != cohort::fingerprint()
                || r["task"] != name
                || r["arm"] != arm
                || r["tier"] != row["tier"]
                || r["status"] != "complete"
            {
                return Err(format!("cell identity mismatch: {id}").into());
            }
            let primary = &r["splits"][r["selection"].as_str().ok_or("selection missing")?];
            let all = r["splits"]
                .as_object()
                .ok_or("splits missing")?
                .values()
                .filter(|s| s["examples"].as_u64().is_some_and(|n| n > 0))
                .all(cohort::passes);
            records.push(
                json!({"task":name,"tier":row["tier"],"arm":arm,"status":"complete",
                "primary_pass":cohort::passes(primary),"all_split_pass":all,
                "primary_accuracy":primary["sequence_accuracy"],
                "train_accuracy":r["splits"]["train"]["sequence_accuracy"],
                "training_ce":r["training_ce"],"fresh_fw_gap":r["fresh_fw_gap"],
                "updates":r["updates"],"stop_reason":r["stop_reason"],
                "seconds":r["training_seconds_including_setup"],"model_sha256":r["model_sha256"]}),
            );
        }
    }
    let actual: BTreeSet<_> = fs::read_to_string(root.join("jobs.tsv"))?
        .lines()
        .map(|line| {
            let (arm, name) = line.split_once('\t').unwrap_or(("invalid", "invalid"));
            format!("{arm}--{name}")
        })
        .collect();
    if actual != expected {
        return Err("job inventory mismatch".into());
    }
    let mut totals = vec![];
    for arm in ["scaled", "calibrated"] {
        for tier in ["short", "long"] {
            let selected: Vec<_> = records
                .iter()
                .filter(|r| r["arm"] == arm && r["tier"] == tier)
                .collect();
            totals.push(json!({"arm":arm,"tier":tier,"planned":selected.len(),
                "complete":selected.iter().filter(|r| r["status"] == "complete").count(),
                "primary_passes":selected.iter().filter(|r| r["primary_pass"] == true).count(),
                "all_split_passes":selected.iter().filter(|r| r["all_split_pass"] == true).count()}));
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({"fingerprint":cohort::fingerprint(),
        "planned":records.len(),"totals":totals,"records":records}))?
    );
    Ok(())
}

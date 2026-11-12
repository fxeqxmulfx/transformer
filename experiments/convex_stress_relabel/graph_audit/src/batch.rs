//! Serial interpretation only after all frozen producer outcomes are sealed.
use crate::{PRODUCER, PROTOCOL, Result, audit, fingerprint, hash};
use serde_json::{Value, json};
use std::{
    fs,
    io::{BufWriter, Write},
    path::Path,
};

fn write_new(path: &Path, value: &Value) -> Result<()> {
    let mut writer = BufWriter::new(fs::File::create_new(path)?);
    serde_json::to_writer(&mut writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

pub fn audit_all(root: &Path, output: &Path) -> Result<Value> {
    let bytes = fs::read(root.join("summary.json"))?;
    let parent: Value = serde_json::from_slice(&bytes)?;
    let protocol: Value = serde_json::from_str(PROTOCOL)?;
    let expected_complete = parent["complete"]
        .as_u64()
        .ok_or("missing completed count")?;
    let expected_rejected = parent["proposal_ineligible"]
        .as_u64()
        .ok_or("missing rejected count")?;
    if parent["source_fingerprint"] != PRODUCER
        || parent["protocol"] != protocol
        || parent["cells"] != 96
        || expected_complete.checked_add(expected_rejected) != Some(96)
        || parent["choices"].as_array().map(Vec::len) != Some(24)
    {
        return Err("complete frozen 96-cell producer summary required".into());
    }
    fs::create_dir(output)?;
    let mut records = vec![];
    let mut complete = 0_u64;
    let mut rejected = 0_u64;
    for task in protocol["tasks"].as_array().unwrap() {
        for arm in protocol["arms"].as_array().unwrap() {
            for coding in protocol["codings"].as_array().unwrap() {
                for budget in protocol["budgets"].as_array().unwrap() {
                    let id = [arm, task, coding, budget]
                        .map(|x| x.as_str().unwrap())
                        .join("--");
                    let record = audit(root, &id)?;
                    complete += u64::from(record["status"] == "complete");
                    rejected += u64::from(record["status"] == "proposal_ineligible");
                    write_new(&output.join(format!("{id}.json")), &record)?;
                    records.push(json!({"id":id,"status":record["status"],
                        "result_sha256":record["result_sha256"],
                        "training_graph_and_completion_conditions":record["training_graph_and_completion_conditions"],
                        "declared_nuclear_bound_le_dimension":record["declared_nuclear_bound_le_dimension"],
                        "observed_ideal_ce_excess":record["observed_ideal_ce_excess"],
                        "completion_comparison":record["completion_comparison"]}));
                }
            }
        }
    }
    if records.len() != 96 || complete != expected_complete || rejected != expected_rejected {
        return Err("audited outcome counts differ from producer summary".into());
    }
    let summary = json!({"auditor_fingerprint":fingerprint(),"producer_fingerprint":PRODUCER,
        "producer_summary_sha256":hash(&bytes),"proof_commit":"dade2361",
        "cells":records.len(),"complete":complete,"proposal_ineligible":rejected,"records":records,
        "changes_training_or_selection":false,"exact_real_arithmetic_certificate":false,"full35_result":false});
    write_new(&output.join("summary.json"), &summary)?;
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn incomplete_or_changed_parent_never_creates_output_and_missing_cells_cannot_summarize()
    -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "graph-audit-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        fs::create_dir(&root)?;
        let output = root.join("output");
        assert!(audit_all(&root, &output).is_err());
        assert!(!output.exists());
        let mut parent = json!({"source_fingerprint":PRODUCER,"protocol":serde_json::from_str::<Value>(PROTOCOL)?,
            "cells":96,"complete":95,"proposal_ineligible":0,"choices":vec![Value::Null;24]});
        let path = root.join("summary.json");
        fs::write(&path, serde_json::to_vec(&parent)?)?;
        assert!(audit_all(&root, &output).is_err());
        assert!(!output.exists());
        parent["complete"] = 96.into();
        parent["source_fingerprint"] = "changed".into();
        fs::write(&path, serde_json::to_vec(&parent)?)?;
        assert!(audit_all(&root, &output).is_err());
        assert!(!output.exists());
        parent["source_fingerprint"] = PRODUCER.into();
        fs::write(&path, serde_json::to_vec(&parent)?)?;
        assert!(audit_all(&root, &output).is_err());
        assert!(output.exists());
        assert!(!output.join("summary.json").exists());
        assert!(
            audit_all(&root, &output)
                .unwrap_err()
                .to_string()
                .contains("File exists")
        );
        fs::remove_dir_all(root)?;
        Ok(())
    }
}

//! Require replay seals and artifact bindings; preserve every process failure.
use super::*;
pub fn summarize(root: &Path) -> Result<Value> {
    let inv = inventory(root)?;
    let mut records = vec![];
    let mut references = vec![];
    let (mut passes, mut all_passes, mut failures) = (0, 0, 0);
    for name in tasks()? {
        let run_exit: i32 = fs::read_to_string(root.join(format!("{name}.run.exit")))?
            .trim()
            .parse()?;
        let replay_exit: Option<i32> = if run_exit == 0 {
            Some(
                fs::read_to_string(root.join(format!("{name}.replay.exit")))?
                    .trim()
                    .parse()?,
            )
        } else {
            None
        };
        if run_exit != 0 || replay_exit != Some(0) {
            failures += 1;
            for mode in ["full_batch", "minibatch", "forward_only"] {
                references.push(json!({"task":name,"mode":mode,
                    "standalone_constructor_reference":true,"gradient_training_run":false,
                    "status":"process_failure","selected":null}));
            }
            records.push(
                json!({"task":name,"status":"process_failure","run_exit":run_exit,
                "replay_exit":replay_exit,"planned_exact_slots":6,"planned_interval_slots":18,
                "partial_artifacts":root.join(&name),
                "run_log_sha256":hash(&bytes(&root.join(format!("{name}.log")))?)}),
            );
            continue;
        }
        let dir = root.join(&name);
        let raw = bytes(&dir.join("result.json"))?;
        let r: Value = serde_json::from_slice(&raw)?;
        let seal: Value = read(&root.join(format!("{name}.replay.json")))?;
        let expected_input = inv["inputs"]
            .as_array()
            .ok_or("missing inputs")?
            .iter()
            .find(|i| i["task"] == name)
            .ok_or("missing original input record")?;
        if r["metadata"]["source_fingerprint"] != fingerprint()
            || r["metadata"]["binary_sha256"] != inv["binary_sha256"]
            || r["metadata"]["producer_commit"] != inv["producer_commit"]
            || r["metadata"]["protocol"] != protocol()?
            || r["metadata"]["input"] != *expected_input
            || read::<Value>(&dir.join("metadata.json"))? != r["metadata"]
            || seal["replayed"] != true
            || seal["full_proposal_state_replay"] != true
            || seal["raw_latent_ce_metrics_audit"] != true
            || seal["task"] != name
            || seal["source_fingerprint"] != fingerprint()
            || seal["result_sha256"] != hash(&raw)
            || seal["choice"] != r["choice"]
            || seal["selection_sha256"] != r["selection_sha256"]
            || r["selection_sha256"] != hash(&bytes(&dir.join("selection.json"))?)
            || read::<Value>(&dir.join("selection.json"))? != r["choice"]
        {
            return Err("missing or changed density replay binding".into());
        }
        for (file, field) in [
            ("input.json", "input_sha256"),
            ("earlier-proposal.json", "earlier_sha256"),
            ("report.json", "report_sha256"),
        ] {
            if r["choice"][field] != hash(&bytes(&dir.join(file))?) {
                return Err("changed density artifact after replay".into());
            }
        }
        let selected = r["evaluation"]["candidates"]
            .as_array()
            .ok_or("missing candidates")?
            .iter()
            .find(|c| c["id"] == r["evaluation"]["selected"]);
        passes += usize::from(selected.is_some_and(|c| c["primary_pass"] == true));
        all_passes += usize::from(selected.is_some_and(|c| c["all_splits_pass"] == true));
        for mode in ["full_batch", "minibatch", "forward_only"] {
            references.push(json!({"task":name,"mode":mode,"standalone_constructor_reference":true,
                "gradient_training_run":false,"selected":selected,"report_path":dir.join("report.json")}));
        }
        records.push(json!({"task":name,"result":r,"seal":seal}));
    }
    Ok(
        json!({"source_fingerprint":fingerprint(),"producer_commit":inv["producer_commit"],
        "protocol":protocol()?,"tasks":records.len(),"primary_passes":passes,"all_split_passes":all_passes,
        "process_failures":failures,"records":records,"mode_references":references,
        "integrated_with_original105":false,"full35_result":false,"fresh_transformer_comparison":false}),
    )
}

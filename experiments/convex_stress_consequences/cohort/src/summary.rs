use super::*;
pub fn summarize(root: &Path) -> Result<Value> {
    let (inv, parent) = inventory::inventory(root)?;
    summarize_with(root, &inv, parent)
}
pub(super) fn summarize_with(root: &Path, inv: &Value, parent: Value) -> Result<Value> {
    parent::validate(&parent)?;
    let mut choices = parent["choices"]
        .as_array()
        .ok_or("missing full parent choices")?
        .clone();
    let (mut records, mut failures, mut seconds) = (vec![], 0, 0.);
    for name in tasks()? {
        let status: i32 = fs::read_to_string(root.join(format!("{name}.run.exit")))?
            .trim()
            .parse()?;
        let replay_status: Option<i32> = if status == 0 {
            Some(
                fs::read_to_string(root.join(format!("{name}.replay.exit")))?
                    .trim()
                    .parse()?,
            )
        } else {
            None
        };
        if status != 0 || replay_status != Some(0) {
            failures += 1;
            records.push(json!({"task":name,"status":"process_failure","run_exit":status,"replay_exit":replay_status,
                "prior_choices_retained":3,"partial_artifacts":root.join(&name),
                "run_log_sha256":digest(&root.join(format!("{name}.log")))?}));
            continue;
        }
        let dir = root.join(&name);
        let result: Value = read(&dir.join("result.json"))?;
        let replay: Value = read(&root.join(format!("{name}.replay.json")))?;
        let input = inv["inputs"]
            .as_array()
            .ok_or("missing input inventory")?
            .iter()
            .find(|r| r["task"] == name)
            .ok_or("missing original record")?;
        if result["metadata"]["input"] != *input
            || result["metadata"]["source_fingerprint"] != fingerprint()
            || result["metadata"]["binary_sha256"] != inv["binary_sha256"]
            || result["metadata"]["producer_commit"] != inv["producer_commit"]
            || result["metadata"]["protocol"] != protocol()?
            || result["metadata"]["parent_sha256"] != inv["parent_sha256"]
            || read::<Value>(&dir.join("metadata.json"))? != result["metadata"]
            || result["ce_gradients"] != 0
            || replay["task"] != name
            || replay["replayed"] != true
            || replay["source_fingerprint"] != fingerprint()
            || replay["parent_sha256"] != inv["parent_sha256"]
            || replay["result_sha256"] != digest(&dir.join("result.json"))?
            || replay["all_view_reports_replayed"] != true
            || replay["raw_latent_ce_metrics_audit"] != true
            || replay["unconditional_ground_truth_identification"] != false
            || replay["selection_sha256"] != result["selection_sha256"]
            || result["selection_sha256"] != digest(&dir.join("selection.json"))?
        {
            return Err("complete result/replay metadata or hashes differ".into());
        }
        let parents = parent::choices(&parent, &name, input)?;
        if read::<Vec<Value>>(&dir.join("parents.json"))? != parents {
            return Err("previous choices changed".into());
        }
        let bank: Bank = read(&dir.join("bank.json"))?;
        for r in &bank.views {
            artifacts::verify(&dir, &r.artifact)?;
        }
        let refs = selection::select(&bank, &parents)?;
        let seal = selection::seal(&dir, &bank, &refs)?;
        if read::<Value>(&dir.join("selection.json"))? != seal
            || result["selection"] != seal
            || replay["references"] != refs
            || result["views"] != serde_json::to_value(&bank.views)?
            || result["candidate_count"] != bank.candidates.len()
        {
            return Err("sealed view artifacts, choices or bank changed after replay".into());
        }
        let updated = result["evaluation"]["choices"]
            .as_array()
            .ok_or("missing mode choices")?;
        if updated.len() != 3 {
            return Err("mode coverage changed".into());
        }
        for mode in modes()? {
            let matching: Vec<_> = updated
                .iter()
                .filter(|r| r["task"] == name && r["training_mode"] == mode)
                .collect();
            if matching.len() != 1 {
                return Err("mode missing or duplicated".into());
            }
            let saved = matching[0];
            let old = choices
                .iter_mut()
                .find(|r| r["task"] == name && r["training_mode"] == mode)
                .ok_or("missing full-suite baseline")?;
            if saved["baseline"] != *old {
                return Err("new choice did not preserve its full parent".into());
            }
            *old = saved.clone();
        }
        seconds += result["construction_seconds"]
            .as_f64()
            .filter(|x| x.is_finite() && *x >= 0.)
            .ok_or("invalid construction duration")?;
        records.push(json!({"task":name,"status":"complete","result":result,"seal":replay}));
    }
    let by_mode: Vec<_> = modes()?.iter().map(|mode| {
        let rows: Vec<_> = choices.iter().filter(|r| r["training_mode"] == *mode).collect();
        json!({"training_mode":mode,"cases":rows.len(),"no_model":rows.iter().filter(|r| r["status"]=="no_model").count(),
            "primary_passes":rows.iter().filter(|r| r["primary_pass"]==true).count(),
            "all_split_passes":rows.iter().filter(|r| r["all_splits_pass"]==true).count()})
    }).collect();
    Ok(
        json!({"source_fingerprint":fingerprint(),"producer_commit":inv["producer_commit"],"protocol":protocol()?,
        "tasks":records.len(),"process_failures":failures,"records":records,"choices_compared":choices.len(),
        "full35_pass":choices.iter().all(|r| r["all_splits_pass"]==true),"choices":choices,"by_mode":by_mode,
        "constructor_accounting":{"planned_constructions":tasks()?.len(),"mode_references":3*tasks()?.len(),
            "completed_seconds":seconds,"charged_once":true,"ce_gradient_evaluations":0,
            "forward_proposals_shared_across_modes":true,"all_prior_costs_retained_in_complete_parent":true},
        "parent_sha256":inv["parent_sha256"],"complete_parent":parent,"full35_result":true,
        "unconditional_ground_truth_identification":false,"fresh_transformer_comparison":false,
        "pending_followups":["retained_depth_integration","fresh_transformer_comparison"]}),
    )
}

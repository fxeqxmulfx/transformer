use super::*;
pub fn summarize(root: &Path) -> Result<Value> {
    let (inv, parent) = inventory(root)?;
    summarize_with(root, &inv, parent)
}
pub(super) fn summarize_with(root: &Path, inv: &Value, parent: Value) -> Result<Value> {
    parent::validate(&parent)?;
    let mut choices = parent["choices"]
        .as_array()
        .ok_or("missing full parent")?
        .clone();
    let mut records = vec![];
    let mut failures = 0;
    let mut seconds = 0.;
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
                "prior_choices_retained":3,"partial_artifacts":root.join(&name),"run_log_sha256":hash(&bytes(&root.join(format!("{name}.log")))?)}));
            continue;
        }
        let dir = root.join(&name);
        let raw = bytes(&dir.join("result.json"))?;
        let result: Value = serde_json::from_slice(&raw)?;
        let seal: Value = read(&root.join(format!("{name}.replay.json")))?;
        let input = inv["inputs"]
            .as_array()
            .ok_or("missing original inventory")?
            .iter()
            .find(|i| i["task"] == name)
            .ok_or("missing original record")?;
        if result["metadata"]["input"] != *input
            || result["metadata"]["source_fingerprint"] != fingerprint()
            || result["metadata"]["binary_sha256"] != inv["binary_sha256"]
            || result["metadata"]["producer_commit"] != inv["producer_commit"]
            || result["metadata"]["protocol"] != protocol()?
            || result["metadata"]["parent_sha256"] != inv["parent_sha256"]
            || read::<Value>(&dir.join("metadata.json"))? != result["metadata"]
            || seal["task"] != name
            || seal["replayed"] != true
            || seal["source_fingerprint"] != fingerprint()
            || seal["parent_sha256"] != inv["parent_sha256"]
            || seal["result_sha256"] != hash(&raw)
            || seal["full_integer_strip_state_replay"] != true
            || seal["raw_latent_ce_metrics_audit"] != true
            || seal["selection_sha256"] != result["selection_sha256"]
            || result["selection_sha256"] != hash(&bytes(&dir.join("selection.json"))?)
            || read::<Value>(&dir.join("selection.json"))? != result["selection"]
            || seal["references"] != result["selection"]["references"]
        {
            return Err("modular complete outcome/replay binding changed".into());
        }
        for (file, key) in [
            ("input.json", "input_sha256"),
            ("parents.json", "parents_sha256"),
            ("report.json", "report_sha256"),
        ] {
            if result["selection"][key] != hash(&bytes(&dir.join(file))?) {
                return Err("modular artifact changed after replay".into());
            }
        }
        let parents = parent::choices(&parent, &name, input)?;
        if read::<Vec<Value>>(&dir.join("parents.json"))? != parents {
            return Err("selected parent rows changed".into());
        }
        let updated = result["evaluation"]["choices"]
            .as_array()
            .ok_or("missing mode choices")?;
        if updated.len() != 3 {
            return Err("modular mode coverage changed".into());
        }
        for mode in modes()? {
            let matching: Vec<_> = updated
                .iter()
                .filter(|r| r["task"] == name && r["training_mode"] == mode)
                .collect();
            if matching.len() != 1 {
                return Err("modular mode duplicated or missing".into());
            }
            let saved = matching[0];
            let old = choices
                .iter_mut()
                .find(|r| r["task"] == name && r["training_mode"] == mode)
                .ok_or("missing full-suite baseline")?;
            if saved["baseline"] != *old {
                return Err("modular update did not preserve its earlier choice".into());
            }
            *old = saved.clone();
        }
        seconds += result["construction_seconds"]
            .as_f64()
            .filter(|x| x.is_finite() && *x >= 0.)
            .ok_or("invalid construction duration")?;
        records.push(json!({"task":name,"status":"complete","result":result,"seal":seal}));
    }
    let by_mode:Vec<_>=modes()?.iter().map(|mode| {
        let rows:Vec<_>=choices.iter().filter(|r|r["training_mode"]==*mode).collect();
        json!({"training_mode":mode,"cases":rows.len(),"no_model":rows.iter().filter(|r|r["status"]=="no_model").count(),
            "primary_passes":rows.iter().filter(|r|r["primary_pass"]==true).count(),"all_split_passes":rows.iter().filter(|r|r["all_splits_pass"]==true).count()})
    }).collect();
    Ok(
        json!({"source_fingerprint":fingerprint(),"producer_commit":inv["producer_commit"],"protocol":protocol()?,
        "tasks":records.len(),"process_failures":failures,"records":records,"choices_compared":choices.len(),
        "full35_pass":choices.iter().all(|r|r["all_splits_pass"]==true),"choices":choices,"by_mode":by_mode,
        "constructor_accounting":{"planned_constructions":tasks()?.len(),"mode_references":3*tasks()?.len(),
            "completed_seconds":seconds,"charged_once":true,"ce_gradient_evaluations":0},
        "parent_sha256":inv["parent_sha256"],"complete_parent":parent,"full35_result":true,
        "fresh_transformer_comparison":false,"pending_followups":["retained_depth_integration","fresh_transformer_comparison"]}),
    )
}

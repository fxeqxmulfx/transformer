//! Keep all35 tasks and both105-choice policies, with actual changed scores.
//! Complete319 and every stronger older reference remain hash-bound inputs.
use super::*;
fn comparisons(current: &Value, reference: &Value) -> Result<Vec<Value>> {
    let mut rows = Vec::new();
    for arm in current["arms"].as_array().ok_or("missing current arms")? {
        let old = reference["arms"]
            .as_array()
            .ok_or("missing reference arms")?
            .iter()
            .find(|a| a["initializer"] == arm["initializer"])
            .ok_or("missing reference policy")?;
        for row in arm["choices"].as_array().ok_or("missing choices")? {
            let before = old["choices"]
                .as_array()
                .ok_or("missing old choices")?
                .iter()
                .find(|r| r["task"] == row["task"] && r["training_mode"] == row["training_mode"])
                .ok_or("missing reference endpoint")?;
            let split = row["primary_split"].as_str().ok_or("missing primary")?;
            if before["primary_split"] != split {
                return Err("reference primary changed".into());
            }
            let a = &before["metrics"][split];
            let b = &row["metrics"][split];
            if a["targets"] != b["targets"] {
                return Err("reference target denominator changed".into());
            }
            rows.push(json!({"initializer":arm["initializer"],"task":row["task"],"mode":row["training_mode"],
                "before_reference":before["reference"],"after_reference":row["reference"],"before_raw_correct":correct(a)?,"after_raw_correct":correct(b)?,
                "targets":b["targets"],"before_raw_accuracy":raw_accuracy(a),"after_raw_accuracy":raw_accuracy(b),
                "before_ce":a["loss"],"after_ce":b["loss"],"before_train_ce":before["train_ce"],"after_train_ce":row["train_ce"],
                "before_original_success":before["primary_pass"],"after_original_success":row["primary_pass"]}));
        }
    }
    if rows.len() != 210 {
        return Err("both105-choice comparisons required".into());
    }
    Ok(rows)
}
pub(crate) fn retain_comparison(summary: &mut Value, comparison: &Value) -> Result<()> {
    counts::summarize(comparison)?;
    summary["changes_vs332"] = json!(comparisons(summary, comparison)?);
    summary["retained332"] = json!({"summary_file":"comparison332.json",
        "summary_sha256":protocol()?["comparison332"]["summary_sha256"],
        "complete_manifest_file":"comparison332-manifest.sha256","artifacts":163,
        "status":"complete reviewed geometry comparison; CE regressions retained"});
    Ok(())
}
pub(crate) fn retain_330(summary: &mut Value, comparison: &Value) -> Result<()> {
    counts::summarize(comparison)?;
    summary["changes_vs330"] = json!(comparisons(summary, comparison)?);
    summary["retained330"] = json!({"summary_file":"comparison330.json",
        "summary_sha256":protocol()?["comparison330"]["summary_sha256"],
        "complete_manifest_file":"comparison330-manifest.sha256","artifacts":156,
        "status":"complete reviewed local-head comparison retained"});
    Ok(())
}
pub(crate) fn combine(mut parent: Value, results: &[Value]) -> Result<Value> {
    counts::summarize(&parent)?;
    if results.len() != 35
        || results
            .iter()
            .map(|r| r["task"].as_str())
            .collect::<BTreeSet<_>>()
            .len()
            != 35
        || results.iter().any(|r| r["status"] != "complete")
    {
        return Err("cannot replace failed or missing cases with old success".into());
    }
    let baseline = json!({"arms":parent["arms"]});
    let mut fresh = BTreeMap::new();
    for result in results {
        for row in result["outcomes"].as_array().ok_or("missing outcomes")? {
            let key = (
                row["task"].as_str().ok_or("missing task")?.to_owned(),
                row["parent_reference"]
                    .as_str()
                    .ok_or("missing old reference")?
                    .to_owned(),
            );
            if fresh.insert(key, row).is_some() {
                return Err("duplicate new endpoint".into());
            }
        }
    }
    let mut used = BTreeSet::new();
    for arm in parent["arms"].as_array_mut().unwrap() {
        let choices = arm["choices"].as_array_mut().unwrap();
        for row in choices.iter_mut() {
            let key = (
                row["task"].as_str().unwrap().to_owned(),
                row["reference"].as_str().unwrap().to_owned(),
            );
            if let Some(new) = fresh.get(&key) {
                used.insert(key.clone());
                row["parent_reference"] = json!(key.1);
                for key in [
                    "reference",
                    "metrics",
                    "primary_pass",
                    "all_splits_pass",
                    "train_ce",
                    "model_sha256",
                    "model_info",
                    "metric_provenance",
                ] {
                    row[key] = new[key].clone();
                }
            } else {
                row["metric_provenance"] = json!({"kind":"inherited_reviewed319","parent_summary_sha256":protocol()?["parent_summary_sha256"]});
            }
        }
        let mut modes = Vec::new();
        for mode in ["full_batch", "minibatch", "forward_only"] {
            let selected: Vec<_> = choices
                .iter()
                .filter(|r| r["training_mode"] == mode)
                .collect();
            if selected.len() != 35 {
                return Err("original mode denominator changed".into());
            }
            modes.push(json!({"training_mode":mode,"cases":35,"primary_passes":selected.iter().filter(|r|r["primary_pass"]==true).count(),
                "all_split_passes":selected.iter().filter(|r|r["all_splits_pass"]==true).count(),
                "fresh_metrics":selected.iter().filter(|r|r["metric_provenance"]["kind"]=="fresh_changed_model").count()}));
        }
        let all = choices.iter().all(|r| r["primary_pass"] == true);
        let held = choices.iter().all(|r| r["all_splits_pass"] == true);
        arm["by_mode"] = json!(modes);
        arm["full35_pass"] = json!(all);
        arm["complete_primary_pass"] = json!(all);
        arm["all_split_diagnostic_pass"] = json!(held);
    }
    if used.len() != fresh.len() {
        return Err("unselected new outcome would disappear".into());
    }
    parent["retained319"] = json!({"summary_file":"baseline.json","summary_sha256":protocol()?["parent_summary_sha256"],
        "complete_manifest_file":"parent-manifest.sha256","artifact_count":331,"arms":baseline["arms"],
        "old_records":parent["records"],"old_changes":parent["changes"],"constructor_accounting":parent["constructor_accounting"],
        "new_optimizer_updates":parent["new_optimizer_updates"],"new_gradient_batches":parent["new_gradient_batches"]});
    parent["records"] = json!(results);
    parent["changes_vs319"] = json!(comparisons(&parent, &baseline)?);
    for (key, label) in [
        ("prior_crossing_reference", "crossing"),
        ("prior_exclusion_reference", "exclusion"),
        ("prior_region_reference", "region"),
        ("prior_local_reference", "guarded"),
    ] {
        parent[format!("changes_vs_{label}_reference")] =
            json!(comparisons(&parent, &parent[key])?);
    }
    parent["changes"] = parent["changes_vs319"].clone();
    parent["primary_raw_totals"] = counts::summarize(&parent)?;
    parent["per_task99"] = goals::summarize(&parent)?;
    parent["new_optimizer_updates"] = json!(
        fresh
            .values()
            .map(|r| r["model_info"]["new_optimizer_updates"].as_u64().unwrap())
            .sum::<u64>()
    );
    parent["new_gradient_batches"] = json!(
        fresh
            .values()
            .map(|r| r["model_info"]["new_gradient_batches"].as_u64().unwrap())
            .sum::<u64>()
    );
    parent["new_local_row_updates"] = json!(0);
    parent["constructor_accounting"] = json!({"new_case_forecasts":"inventory.json","parent_costs":"retained319.constructor_accounting",
        "all_calls_checked":true,"old_constructions_reserved":2,"new_training_passes_reserved":3});
    parent["source_fingerprint"] = json!(fingerprint());
    Ok(parent)
}
pub fn summarize(root: &Path) -> Result<Value> {
    let inv = inputs::inventory(root)?;
    if digest(&root.join("baseline.json"))? != inv["protocol"]["parent_summary_sha256"] {
        return Err("complete319 snapshot changed".into());
    }
    let mut results = Vec::new();
    let mut failures = Vec::new();
    for case in inv["cases"].as_array().ok_or("missing cases")? {
        let task = name(case["task"].as_str().ok_or("missing task")?)?;
        let success = ["run", "replay"].iter().all(|kind| {
            fs::read_to_string(root.join(format!("{task}.{kind}.exit")))
                .is_ok_and(|s| s.trim() == "0")
        });
        if !success {
            failures.push(task.to_owned());
            continue;
        }
        let result = root.join(task).join("result.json");
        let replay: Value = read(&root.join(format!("{task}.replay.json")))?;
        if replay["result_sha256"] != digest(&result)? {
            return Err("replayed result changed".into());
        }
        results.push(read(&result)?);
    }
    if !failures.is_empty() {
        return Ok(
            json!({"complete":false,"tasks":35,"choices_per_arm":105,"failed_cases":failures,
        "completed_cases":results,"per_task99":null,"reason":"failure stays in35 denominator; complete success unavailable","inventory_sha256":digest(&root.join("inventory.json"))?}),
        );
    }
    let mut summary = combine(read(&root.join("baseline.json"))?, &results)?;
    if digest(&root.join("comparison332.json"))? != protocol()?["comparison332"]["summary_sha256"] {
        return Err("retained complete332 snapshot changed".into());
    }
    retain_comparison(&mut summary, &read(&root.join("comparison332.json"))?)?;
    if digest(&root.join("comparison330.json"))? != protocol()?["comparison330"]["summary_sha256"] {
        return Err("retained complete330 snapshot changed".into());
    }
    retain_330(&mut summary, &read(&root.join("comparison330.json"))?)?;
    summary["complete"] = json!(true);
    summary["producer_commit"] = inv["request"]["producer_commit"].clone();
    summary["inventory_sha256"] = json!(digest(&root.join("inventory.json"))?);
    summary["parent_summary_sha256"] = inv["protocol"]["parent_summary_sha256"].clone();
    summary["prior_refused_preparation"] = inv["protocol"]["previous_refused_preparation"].clone();
    summary["historical_diff_gpt"] = json!({"file":"diff-gpt-history.json","sha256":digest(&root.join("diff-gpt-history.json"))?,
        "source":"5a37c516339f23037d191b0659a5a0853984328d","seeds":[0,1,2],"historical_full35_success":true,"fresh_matched_budget_comparison":false});
    Ok(summary)
}

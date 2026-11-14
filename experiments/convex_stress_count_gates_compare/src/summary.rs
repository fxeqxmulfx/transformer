//! All35 original tasks and both105-choice policies remain in the denominator.
//! New raw metrics replace only the frozen eligible references. All previous
//! constructor costs and every conventional/depth endpoint remain explicit.
use super::*;

pub(crate) fn combine(mut parent: Value, results: &[Value], parent_pin: &Value) -> Result<Value> {
    if parent["tasks"] != 35
        || parent["choices_per_arm"] != 105
        || parent["process_failures"] != 0
        || parent["baselines"]["transformer_endpoints"]
            .as_array()
            .map(Vec::len)
            != Some(35)
        || parent["baselines"]["depth_outcomes"]
            .as_array()
            .map(Vec::len)
            != Some(336)
    {
        return Err("complete original scope and baselines required".into());
    }
    if results.len() != 35
        || results
            .iter()
            .map(|r| r["task"].as_str())
            .collect::<BTreeSet<_>>()
            .len()
            != 35
    {
        return Err("all35 distinct original results required".into());
    }
    let mut fresh = BTreeMap::new();
    for result in results {
        if result["status"] != "complete" {
            return Err("failed original case cannot inherit success".into());
        }
        for row in result["outcomes"]
            .as_array()
            .ok_or("missing new outcomes")?
        {
            let key = (
                row["task"].as_str().ok_or("missing task")?.to_owned(),
                row["parent_reference"]
                    .as_str()
                    .ok_or("missing reference")?
                    .to_owned(),
            );
            if fresh.insert(key, row).is_some() {
                return Err("duplicate fresh model outcome".into());
            }
        }
    }
    let mut changes = Vec::new();
    let arms = parent["arms"]
        .as_array_mut()
        .ok_or("missing original arms")?;
    if arms.len() != 2 {
        return Err("both initializer policies required".into());
    }
    if arms
        .iter()
        .map(|a| a["initializer"].as_str())
        .collect::<BTreeSet<_>>()
        != BTreeSet::from([Some("minimum_ce"), Some("first_small_ce")])
    {
        return Err("both distinct original initializer policies required".into());
    }
    let tasks = results
        .iter()
        .map(|r| r["task"].as_str())
        .collect::<BTreeSet<_>>();
    let mut used = BTreeSet::new();
    for arm in arms.iter_mut() {
        let initializer = arm["initializer"].clone();
        let choices = arm["choices"]
            .as_array_mut()
            .ok_or("missing original choices")?;
        if choices.len() != 105 {
            return Err("all105 original choices required".into());
        }
        let mut seen = BTreeSet::new();
        for choice in choices.iter_mut() {
            let task = choice["task"]
                .as_str()
                .ok_or("missing chosen task")?
                .to_owned();
            if !tasks.contains(&Some(task.as_str())) {
                return Err("choice names a task outside the original results".into());
            }
            let mode = choice["training_mode"]
                .as_str()
                .ok_or("missing chosen mode")?
                .to_owned();
            if !seen.insert((task.clone(), mode)) {
                return Err("duplicate task/mode choice".into());
            }
            let reference = choice["reference"]
                .as_str()
                .ok_or("missing prior reference")?
                .to_owned();
            if let Some(row) = fresh.get(&(task.clone(), reference.clone())) {
                if row["training_mode"] != choice["training_mode"] {
                    return Err("fresh model training mode mismatch".into());
                }
                used.insert((task, reference.clone()));
                let primary = choice["primary_split"]
                    .as_str()
                    .ok_or("missing primary split")?;
                let old = choice["metrics"][primary].clone();
                let new = row["metrics"][primary].clone();
                changes.push(json!({"initializer":initializer,"task":choice["task"],"training_mode":choice["training_mode"],
                    "before_primary_pass":choice["primary_pass"],"after_primary_pass":row["primary_pass"],
                    "before_raw_accuracy":raw_accuracy(&old),"after_raw_accuracy":raw_accuracy(&new),
                    "before_ce":old["loss"],"after_ce":new["loss"],"before_primary_accuracy":old["sequence_accuracy"],
                    "after_primary_accuracy":new["sequence_accuracy"]}));
                choice["parent_reference"] = json!(reference);
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
                    choice[key] = row[key].clone();
                }
            } else {
                choice["metric_provenance"] = json!({"kind":"inherited_reviewed_293","parent_reference":reference,"parent_summary_sha256":parent_pin});
            }
        }
        let mut by_mode = Vec::new();
        for mode in ["full_batch", "minibatch", "forward_only"] {
            let rows: Vec<_> = choices
                .iter()
                .filter(|r| r["training_mode"] == mode)
                .collect();
            if rows.len() != 35 {
                return Err("original mode denominator is not35".into());
            }
            by_mode.push(json!({"training_mode":mode,"cases":rows.len(),"primary_passes":rows.iter().filter(|r|r["primary_pass"]==true).count(),
                "all_split_passes":rows.iter().filter(|r|r["all_splits_pass"]==true).count(),
                "fresh_metrics":rows.iter().filter(|r|r["metric_provenance"]["kind"]=="fresh_changed_model").count()}));
        }
        let full = choices.iter().all(|r| r["primary_pass"] == true);
        let aux = choices.iter().all(|r| r["all_splits_pass"] == true);
        arm["by_mode"] = json!(by_mode);
        arm["full35_pass"] = json!(full);
        arm["complete_primary_pass"] = json!(full);
        arm["all_split_diagnostic_pass"] = json!(aux);
    }
    if used.len() != fresh.len() {
        return Err("unselected fresh outcomes would disappear".into());
    }
    parent["changes"] = json!(changes);
    parent["records"] = json!(results);
    parent["parent_summary_sha256"] = parent_pin.clone();
    parent["source_fingerprint"] = json!(fingerprint());
    let sum_work = |key: &str| -> Result<u64> {
        fresh.values().try_fold(0u64, |total, row| {
            Ok(total
                + row["model_info"][key]
                    .as_u64()
                    .ok_or("missing actual mixture work")?)
        })
    };
    parent["new_optimizer_updates"] = json!(sum_work("new_optimizer_updates")?);
    parent["new_gradient_batches"] = json!(sum_work("new_gradient_batches")?);
    parent["new_local_row_updates"] = json!(sum_work("new_local_row_updates")?);
    parent["raw_label_identification_certified"] = json!(false);
    Ok(parent)
}

pub fn summarize(root: &Path) -> Result<Value> {
    let inv = parent::inventory(root)?;
    let mut results = Vec::new();
    for task in inv["tasks"]
        .as_array()
        .ok_or("missing complete task list")?
    {
        let task = name(task.as_str().ok_or("invalid task")?)?;
        for kind in ["run", "replay"] {
            if fs::read_to_string(root.join(format!("{task}.{kind}.exit")))?.trim() != "0" {
                return Err(
                    "original process or replay failure retained; complete summary refused".into(),
                );
            }
        }
        let replay: Value = read(&root.join(format!("{task}.replay.json")))?;
        let path = root.join(task).join("result.json");
        if replay["result_sha256"] != digest(&path)? {
            return Err("replayed result artifact changed".into());
        }
        results.push(read(&path)?);
    }
    let parent: Value = read(&root.join("parent-snapshot.json"))?;
    let mut summary = combine(parent, &results, &inv["protocol"]["parent_summary_sha256"])?;
    let reference: Value = read(&root.join("local-reference.json"))?;
    reference::validate(&reference)?;
    summary["prior_local_reference"] = reference;
    summary["inventory_sha256"] = json!(digest(&root.join("inventory.json"))?);
    summary["producer_commit"] = inv["request"]["producer_commit"].clone();
    Ok(summary)
}

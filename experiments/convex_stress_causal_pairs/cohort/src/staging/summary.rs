use super::*;
fn exit(root: &Path, file: &str) -> Result<i32> {
    Ok(fs::read_to_string(root.join(file))?.trim().parse()?)
}
/// Bind all actual case files and process evidence, including incomplete cases.
pub(crate) fn case_record(root: &Path, task: &str, inv: &Value) -> Result<Value> {
    name(task)?;
    let run_exit = exit(root, &format!("{task}.run.exit"))?;
    let replay_exit = if run_exit == 0 {
        Some(exit(root, &format!("{task}.replay.exit"))?)
    } else {
        None
    };
    let mut artifacts = vec![];
    for suffix in ["run.exit", "run.resources", "log"] {
        artifacts.push(parent::evidence(&root.join(format!("{task}.{suffix}")))?);
    }
    if replay_exit.is_some() {
        for suffix in [
            "replay.exit",
            "replay.resources",
            "replay.json",
            "replay.log",
        ] {
            artifacts.push(parent::evidence(&root.join(format!("{task}.{suffix}")))?);
        }
    }
    if root.join(task).exists() {
        parent::tree(&root.join(task), &mut artifacts)?;
    }
    let complete = run_exit == 0 && replay_exit == Some(0);
    let result = if complete {
        let result = retained(root, task, inv)?;
        if read::<Value>(&root.join(format!("{task}.replay.json")))? != seal(root, task)? {
            return Err("staging replay seal differs".into());
        }
        Some(result)
    } else {
        None
    };
    Ok(
        json!({"task":task,"status":if complete {"complete"} else {"process_failure"},
        "run_exit":run_exit,"replay_exit":replay_exit,"artifacts":artifacts,"result":result}),
    )
}
pub fn summarize(root: &Path) -> Result<Value> {
    summarize_with(root, &inventory(root)?)
}
pub(crate) fn summarize_with(root: &Path, inv: &Value) -> Result<Value> {
    let tasks: Vec<String> = serde_json::from_value(inv["tasks"].clone())?;
    let records = tasks
        .iter()
        .map(|t| case_record(root, t, inv))
        .collect::<Result<Vec<_>>>()?;
    let completed = records.iter().filter(|r| r["status"] == "complete").count();
    let mut seconds = 0.;
    let mut gradients = 0u64;
    for r in &records {
        if r["status"] == "complete" {
            seconds += r["result"]["construction_seconds"]
                .as_f64()
                .ok_or("missing staged time")?;
            gradients += r["result"]["gradient_batches"]
                .as_u64()
                .ok_or("missing staged work")?;
        }
    }
    if !seconds.is_finite() {
        return Err("nonfinite staged construction total".into());
    }
    Ok(
        json!({"phase":"training_only","source_fingerprint":fingerprint(),
        "inventory_sha256":digest(&root.join("inventory.json"))?,"producer_commit":inv["producer_commit"],
        "binary_sha256":inv["binary_sha256"],"staging_protocol":stage_protocol()?,"model_protocol":protocol()?,
        "tasks":tasks,"inputs":inv["inputs"],"records":records,
        "preparation_resources":parent::evidence(&root.join("prepare.resources"))?,
        "constructor_accounting":{"planned_cases":35,"completed_cases":completed,"process_failures":35-completed,
        "complete_constructor_seconds":seconds,"complete_gradient_batches":gradients,
        "partial_costs_in_retained_resources":completed!=35,"charged_once_across_both_arms":true},
        "heldout_evaluation":false,"parent_comparison":false,"unseen_identification":false}),
    )
}

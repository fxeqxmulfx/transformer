use super::*;
pub(crate) fn case_record(
    root: &Path,
    task: &str,
    inv: &Value,
    x: inputs::Inputs,
) -> Result<Value> {
    name(task)?;
    let run_exit = source::exit(root, &format!("{task}.run.exit"))?;
    let replay_exit = if run_exit == 0 {
        Some(source::exit(root, &format!("{task}.replay.exit"))?)
    } else {
        None
    };
    let mut artifacts = vec![];
    source::tree(&root.join(task), &mut artifacts)?;
    for suffix in ["run.exit", "run.resources", "log"] {
        artifacts.push(source::evidence(&root.join(format!("{task}.{suffix}")))?);
    }
    if replay_exit.is_some() {
        for suffix in [
            "replay.exit",
            "replay.resources",
            "replay.json",
            "replay.log",
        ] {
            artifacts.push(source::evidence(&root.join(format!("{task}.{suffix}")))?);
        }
    }
    let complete = run_exit == 0 && replay_exit == Some(0);
    let result = if complete {
        let result = stage::record(root, task, inv, &x, false)?;
        if read::<Value>(&root.join(format!("{task}.replay.json")))? != stage::seal(root, task)? {
            return Err("refined complete case has altered replay evidence".into());
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
    let inv = inventory(root)?;
    let suite = suite()?;
    let records = planned()?
        .iter()
        .map(|t| case_record(root, t, &inv, inputs::build(&suite, t)?))
        .collect::<Result<Vec<_>>>()?;
    render(root, &inv, records)
}
pub(crate) fn render(root: &Path, inv: &Value, records: Vec<Value>) -> Result<Value> {
    let planned: Vec<String> = serde_json::from_value(inv["planned"].clone())?;
    if records.len() != planned.len() || records.iter().zip(&planned).any(|(r, t)| r["task"] != *t)
    {
        return Err("refined summary lost or reordered a task".into());
    }
    let completed = records.iter().filter(|r| r["status"] == "complete").count();
    let (mut seconds, mut gradients, mut stages) = (0_f64, 0_u64, 0_usize);
    let mut available = [0_usize; 3];
    for record in &records {
        if record["status"] != "complete" {
            continue;
        }
        for stage in record["result"]["stages"]
            .as_array()
            .ok_or("missing completed stage ledger")?
        {
            stages += 1;
            seconds += stage["retained"]["seconds"]
                .as_f64()
                .ok_or("missing constructor time")?;
            if let Some(endpoints) = stage["retained"]["endpoints"].as_array() {
                if endpoints.len() != 3 {
                    return Err("refined summary lost a mode".into());
                }
                for (mode, endpoint) in endpoints.iter().enumerate() {
                    if endpoint["training"].is_object() {
                        available[mode] += 1;
                        gradients += endpoint["training"]["work"]["gradient_batches"]
                            .as_u64()
                            .ok_or("missing derivative count")?;
                    }
                }
            }
        }
    }
    if !seconds.is_finite() || seconds < 0. {
        return Err("invalid refined constructor total".into());
    }
    Ok(
        json!({"phase":"training_only","source_fingerprint":fingerprint(),
        "inventory_sha256":digest(&root.join("inventory.json"))?,"request":inv["request"],
        "binary_sha256":inv["binary_sha256"],"protocol":protocol()?,"tasks":inv["tasks"],"planned":planned,
        "inputs":inv["inputs"],"source_cases":inv["discoveries"],"records":records,
        "preparation_resources":source::evidence(&root.join("prepare.resources"))?,
        "constructor_accounting":{"planned_cases":planned.len(),"completed_cases":completed,
            "process_failures":planned.len()-completed,"complete_stages":stages,
            "complete_constructor_seconds":seconds,"complete_gradient_batches":gradients,
            "available_endpoints_by_mode":available,"new_discovery_work":0,
            "source_costs_retained_in_source_cases":true,"partial_costs_in_retained_resources":completed!=planned.len(),
            "charged_once_across_both_arms":true},
        "heldout_evaluation":false,"parent_comparison":false,"unseen_identification":false}),
    )
}

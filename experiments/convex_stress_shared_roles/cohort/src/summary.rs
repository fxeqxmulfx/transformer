//! Complete outcomes/resources, ported readout_kernel/cohort at 5d4eee84 for CM40.
//! No parent selector or held-out evaluator is present in this training stage.
use super::*;
pub(crate) fn case_record(
    root: &Path,
    task: &str,
    inv: &Value,
    x: inputs::Inputs,
) -> Result<Value> {
    name(task)?;
    inputs::check(inv, task, &x)?;
    let run_exit = evidence::exit(root, &format!("{task}.run.exit"))?;
    let replay_exit = if run_exit == 0 {
        Some(evidence::exit(root, &format!("{task}.replay.exit"))?)
    } else {
        None
    };
    let mut artifacts = vec![];
    evidence::tree(&root.join(task), &mut artifacts)?;
    for suffix in ["run.exit", "run.resources", "log"] {
        artifacts.push(evidence::evidence(&root.join(format!("{task}.{suffix}")))?);
    }
    if replay_exit.is_some() {
        for suffix in [
            "replay.exit",
            "replay.resources",
            "replay.json",
            "replay.log",
        ] {
            artifacts.push(evidence::evidence(&root.join(format!("{task}.{suffix}")))?);
        }
    }
    let complete = run_exit == 0 && replay_exit == Some(0);
    let result = if complete {
        let result = stage::record(root, task, inv, &x, false)?;
        if read::<Value>(&root.join(format!("{task}.replay.json")))? != stage::seal(root, task)? {
            return Err("kernel case replay evidence changed".into());
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
        return Err("kernel summary lost or reordered a task".into());
    }
    let completed = records.iter().filter(|r| r["status"] == "complete").count();
    let (mut discovery_seconds, mut head_seconds, mut gradients, mut stages, mut codes) =
        (0_f64, 0_f64, 0_u64, 0_usize, 0_usize);
    let mut available = [0_usize; 3];
    let mut expansions = 0;
    for record in &records {
        if record["status"] != "complete" {
            continue;
        }
        for stage in record["result"]["stages"]
            .as_array()
            .ok_or("missing kernel completed stages")?
        {
            let r = &stage["retained"];
            stages += 1;
            discovery_seconds += r["discovery_seconds"]
                .as_f64()
                .ok_or("missing discovery time")?;
            head_seconds += r["head_seconds"].as_f64().ok_or("missing head time")?;
            codes += usize::from(r["discovery"]["restricted"]["dimension"].is_number());
            expansions += usize::from(r["discovery"]["dimension"].is_number());
            if let Some(endpoints) = r["endpoints"].as_array() {
                if endpoints.len() != 3 {
                    return Err("kernel summary lost a mode".into());
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
    if [discovery_seconds, head_seconds]
        .iter()
        .any(|s| !s.is_finite() || *s < 0.)
    {
        return Err("invalid kernel accumulated times".into());
    }
    Ok(
        json!({"phase":"training_only","source_fingerprint":fingerprint(),
        "inventory_sha256":digest(&root.join("inventory.json"))?,"request":inv["request"],
        "binary_sha256":inv["binary_sha256"],"protocol":protocol()?,"tasks":inv["tasks"],"planned":planned,
        "inputs":inv["inputs"],"records":records,"preparation_resources":evidence::evidence(&root.join("prepare.resources"))?,
        "constructor_accounting":{"planned_cases":planned.len(),"completed_cases":completed,
            "process_failures":planned.len()-completed,"complete_stages":stages,
            "complete_restricted_kernels":codes,"admitted_raw_expansions":expansions,
            "complete_discovery_seconds":discovery_seconds,"complete_head_seconds":head_seconds,
            "complete_gradient_batches":gradients,"available_endpoints_by_mode":available,
            "all_discovery_work_in_stage_records":true,"charged_once_across_both_arms":true,
            "partial_costs_in_retained_resources":completed!=planned.len()},
        "heldout_evaluation":false,"parent_comparison":false,"raw_label_identification":false,
        "target_readout_compatibility_proved":false,"declared_readout_used_in_discovery":true,
        "shared_role_target_proved":false,"shared_role_constraints_used":true,
        "original_record_multiplicity_retained":true,
        "original_raw_ce_unchanged":true}),
    )
}

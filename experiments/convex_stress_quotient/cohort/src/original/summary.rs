//! Preserve complete parents and failed attempts; derive pass flags from original metrics.
use super::*;
fn exit(root: &Path, file: &str) -> Result<i32> {
    Ok(fs::read_to_string(root.join(file))?.trim().parse()?)
}
fn evidence(path: &Path) -> Result<Value> {
    Ok(json!({"path":path,"sha256":digest(path)?,"bytes":fs::metadata(path)?.len()}))
}
fn partial(path: &Path, files: &mut Vec<Value>) -> Result<()> {
    if !path.exists() {
        return Ok(());
    }
    let mut entries: Vec<_> = fs::read_dir(path)?.collect::<std::result::Result<_, _>>()?;
    entries.sort_by_key(|e| e.path());
    for entry in entries {
        if entry.file_type()?.is_dir() {
            partial(&entry.path(), files)?;
        } else {
            files.push(evidence(&entry.path())?);
        }
    }
    Ok(())
}
pub fn summarize(root: &Path) -> Result<Value> {
    let (inv, parent) = inventory::inventory(root, true)?;
    summarize_with(root, &inv, parent, &planned()?)
}
pub(super) fn summarize_with(
    root: &Path,
    inv: &Value,
    parent: Value,
    planned: &[String],
) -> Result<Value> {
    let mut records = vec![];
    let mut offers = vec![];
    let mut evaluated = BTreeMap::new();
    let (mut failures, mut seconds, mut gradients) = (0usize, 0., 0u64);
    let all: Vec<String> = serde_json::from_value(inv["tasks"].clone())?;
    for task in planned {
        name(task)?;
        let code = exit(root, &format!("{task}.run.exit"))?;
        let replay_code = if code == 0 {
            Some(exit(root, &format!("{task}.replay.exit"))?)
        } else {
            None
        };
        let resources = evidence(&root.join(format!("{task}.run.resources")))?;
        if code != 0 || replay_code != Some(0) {
            failures += 1;
            let mut files = vec![];
            partial(&root.join(task), &mut files)?;
            let replay_evidence = if replay_code.is_some() {
                Some(
                    ["json", "log", "resources"]
                        .iter()
                        .map(|suffix| evidence(&root.join(format!("{task}.replay.{suffix}"))))
                        .collect::<Result<Vec<_>>>()?,
                )
            } else {
                None
            };
            records.push(json!({"task":task,"status":"process_failure","run_exit":code,"replay_exit":replay_code,
                "prior_choices_retained":6,"partial_artifacts":files,"run_resources":resources,
                "failed_replay_evidence":replay_evidence,
                "run_log":evidence(&root.join(format!("{task}.log")))?}));
            continue;
        }
        let dir = root.join(task);
        let result: Value = read(&dir.join("result.json"))?;
        let replay: Value = read(&root.join(format!("{task}.replay.json")))?;
        let (config, ledger) = crate::execute::retained(&dir.join("catalogue"))?;
        candidates::verify_stage_bytes(&dir.join("catalogue"), &ledger)?;
        let input = inv["inputs"]
            .as_array()
            .ok_or("missing input inventory")?
            .iter()
            .find(|r| r["task"] == *task)
            .ok_or("missing original input")?;
        let current = candidates::offers(task, &ledger)?;
        let selection = comparison::select(&parent, &current, &all)?;
        let seal = execute::seal(&dir, &selection)?;
        if read::<Value>(&dir.join("metadata.json"))? != result["metadata"]
            || result["metadata"]["input"] != *input
            || result["metadata"]["source_fingerprint"] != fingerprint()
            || result["metadata"]["binary_sha256"] != inv["binary_sha256"]
            || result["metadata"]["producer_commit"] != inv["producer_commit"]
            || result["metadata"]["protocol"] != protocol()?
            || result["metadata"]["parent_summary_sha256"]
                != inv["request"]["parent"]["summary_sha256"]
            || input["configuration_sha256"] != hash(&serde_json::to_vec(&config)?)
            || read::<Vec<comparison::Offer>>(&dir.join("offers.json"))? != current
            || read::<Value>(&dir.join("selection.json"))? != seal
            || result["selection"] != seal
            || result["selection_sha256"] != digest(&dir.join("selection.json"))?
            || result["stages"] != serde_json::to_value(&ledger.stages)?
            || result["eligible_offers"] != current.len()
            || result["all55_stages_retained"] != true
            || result["raw_label_identification"] != false
            || result["total_seconds"]
                .as_f64()
                .is_none_or(|v| !v.is_finite() || v < 0.)
            || replay["task"] != *task
            || replay["replayed"] != true
            || replay["source_fingerprint"] != fingerprint()
            || replay["result_sha256"] != digest(&dir.join("result.json"))?
            || replay["selection_sha256"] != result["selection_sha256"]
            || replay["parent_summary_sha256"] != inv["request"]["parent"]["summary_sha256"]
            || replay["catalogue"]["ledger_sha256"] != digest(&dir.join("catalogue/ledger.json"))?
            || replay["catalogue"]["all_original_stage_reports_replayed"] != true
            || replay["catalogue"]["source_fingerprint"] != fingerprint()
            || replay["catalogue"]["replayed"] != true
            || replay["catalogue"]["stages"] != 55
            || replay["catalogue"]["selected_stages"]
                != serde_json::to_value(ledger.selected_stages)?
            || replay["raw_label_identification"] != false
            || replay["target_readout_compatibility_proved"] != false
            || replay["all_actual_models_and_metrics_replayed"] != true
            || replay["all_ce_steps_replayed"] != true
        {
            return Err(
                "original successful result, parent, catalogue or replay seal changed".into(),
            );
        }
        let duration = ledger.seconds_per_stage.iter().sum::<f64>();
        if result["construction_seconds"] != duration || !duration.is_finite() || duration < 0. {
            return Err("actual constructor timing ledger changed".into());
        }
        seconds += duration;
        let outcomes = result["evaluation"]
            .as_array()
            .ok_or("missing actual model outcomes")?;
        candidates::check_evaluations(task, input, &ledger, outcomes)?;
        for value in outcomes {
            let reference = value["reference"]
                .as_str()
                .ok_or("missing candidate reference")?;
            if value["task"] != *task
                || evaluated
                    .insert(reference.to_string(), value.clone())
                    .is_some()
            {
                return Err("unknown or duplicated model evaluation".into());
            }
            let (primary, all) = passes(
                &value["metrics"],
                value["primary_split"]
                    .as_str()
                    .ok_or("missing primary split")?,
            )?;
            if value["primary_pass"] != primary || value["all_splits_pass"] != all {
                return Err("original success flags differ".into());
            }
        }
        for stage in &ledger.stages {
            for e in stage.retained["endpoints"]
                .as_array()
                .ok_or("missing retained endpoints")?
            {
                gradients += e["training_work"]["gradient_batches"]
                    .as_u64()
                    .ok_or("missing actual gradient work")?;
            }
        }
        offers.extend(current);
        records.push(json!({"task":task,"status":"complete","result":result,"replay":replay,
            "run_resources":resources,"replay_resources":evidence(&root.join(format!("{task}.replay.resources")))?}));
    }
    let selection = comparison::select(&parent, &offers, &all)?;
    let arms = render(&parent, &selection, &evaluated)?;
    Ok(
        json!({"source_fingerprint":fingerprint(),"producer_commit":inv["producer_commit"],"protocol":protocol()?,
        "parent_sha256":inv["request"]["parent"]["summary_sha256"],"complete_parent":parent,
        "parent_snapshots":inv["parent_snapshots"],"records":records,"tasks":planned.len(),"process_failures":failures,
        "selection":selection,"arms":arms,"choose_between_initializers":false,"choices_per_arm":105,
        "all35_transformer_endpoints_retained":true,"unconditional_identification":false,
        "constructor_accounting":{"planned_cases":planned.len(),"completed_cases":planned.len()-failures,
            "complete_stage_constructions":55*(planned.len()-failures),"complete_constructor_seconds":seconds,
            "complete_gradient_batches":gradients,"partial_costs_in_retained_resources":failures>0,
            "charged_once_across_both_arms":true,"all_prior_costs_retained_in_complete_parent":true}}),
    )
}
fn render(
    parent: &Value,
    selection: &Value,
    evaluations: &BTreeMap<String, Value>,
) -> Result<Vec<Value>> {
    let mut arms = vec![];
    for arm in selection["arms"]
        .as_array()
        .ok_or("missing selected arms")?
    {
        let prior = parent["arms"]
            .as_array()
            .ok_or("missing parent arms")?
            .iter()
            .find(|a| a["initializer"] == arm["initializer"])
            .ok_or("missing policy")?;
        let mut choices = vec![];
        for r in arm["choices"].as_array().ok_or("missing choices")? {
            let old = prior["choices"]
                .as_array()
                .ok_or("missing old choices")?
                .iter()
                .find(|o| o["task"] == r["task"] && o["training_mode"] == r["training_mode"])
                .ok_or("missing prior case")?;
            let chosen = if r["selected"] == old["reference"] {
                old.clone()
            } else {
                let reference = r["selected"].as_str().ok_or("existing parent discarded")?;
                let new = evaluations
                    .get(reference)
                    .ok_or("selected actual model absent")?;
                if new["task"] != r["task"] || new["training_mode"] != r["training_mode"] {
                    return Err("selected model task/mode changed".into());
                }
                json!({"task":r["task"],"training_mode":r["training_mode"],"status":"selected","reference":reference,
                    "train_ce":new["train_ce"],"primary_split":new["primary_split"],"metrics":new["metrics"],
                    "primary_pass":new["primary_pass"],"all_splits_pass":new["all_splits_pass"],"saved_endpoint":new,"baseline":old})
            };
            choices.push(chosen);
        }
        let by_mode:Vec<_>=["full_batch","minibatch","forward_only"].iter().map(|mode|{
            let rows:Vec<_>=choices.iter().filter(|r|r["training_mode"]==*mode).collect();
            json!({"training_mode":mode,"cases":rows.len(),"primary_passes":rows.iter().filter(|r|r["primary_pass"]==true).count(),
                "all_split_passes":rows.iter().filter(|r|r["all_splits_pass"]==true).count(),"no_model":rows.iter().filter(|r|r["status"]=="no_model").count()})
        }).collect();
        arms.push(json!({"initializer":arm["initializer"],"full35_pass":choices.iter().all(|r|r["all_splits_pass"]==true),"choices":choices,"by_mode":by_mode}));
    }
    Ok(arms)
}

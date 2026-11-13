//! Complete task scope, failed processes and inherited costs remain visible.
use super::*;
fn exit(root: &Path, file: &str) -> Result<i32> {
    Ok(fs::read_to_string(root.join(file))?.trim().parse()?)
}
pub fn summarize(root: &Path) -> Result<Value> {
    let (inv, parent) = inventory::inventory(root, true)?;
    summarize_with(root, &inv, parent)
}
pub(crate) fn summarize_with(root: &Path, inv: &Value, parent: Value) -> Result<Value> {
    let all: Vec<String> = serde_json::from_value(inv["tasks"].clone())?;
    let (mut records, mut offers, mut evaluated) = (vec![], vec![], BTreeMap::new());
    let (mut failures, mut seconds, mut gradients) = (0usize, 0f64, 0usize);
    for task in &all {
        name(task)?;
        let code = exit(root, &format!("{task}.run.exit"))?;
        let replay_code = if code == 0 {
            Some(exit(root, &format!("{task}.replay.exit"))?)
        } else {
            None
        };
        let resources = parent::evidence(&root.join(format!("{task}.run.resources")))?;
        if code != 0 || replay_code != Some(0) {
            failures += 1;
            let mut partial = vec![];
            if root.join(task).exists() {
                parent::tree(&root.join(task), &mut partial)?;
            }
            let replay_evidence = if replay_code.is_some() {
                Some(
                    ["json", "log", "resources"]
                        .iter()
                        .map(|suffix| {
                            parent::evidence(&root.join(format!("{task}.replay.{suffix}")))
                        })
                        .collect::<Result<Vec<_>>>()?,
                )
            } else {
                None
            };
            records.push(json!({"task":task,"status":"process_failure","run_exit":code,"replay_exit":replay_code,
                "prior_choices_retained":6,"partial_artifacts":partial,"run_resources":resources,
                "failed_replay_evidence":replay_evidence,"run_log":parent::evidence(&root.join(format!("{task}.log")))?}));
            continue;
        }
        let dir = root.join(task);
        let result = execute::retained(root, task, inv, &parent)?;
        let replay: Value = read(&root.join(format!("{task}.replay.json")))?;
        let attempt: execute::Attempt = read(&dir.join("attempt.json"))?;
        if replay["task"] != *task
            || replay["replayed"] != true
            || replay["source_fingerprint"] != fingerprint()
            || replay["parent_summary_sha256"] != inv["request"]["parent"]["summary_sha256"]
            || replay["result_sha256"] != digest(&dir.join("result.json"))?
            || replay["report_sha256"] != digest(&dir.join("attempt.json"))?
            || replay["selection_sha256"] != digest(&dir.join("selection.json"))?
            || replay["actual_complex_states_replayed"] != true
            || replay["all_ce_steps_replayed"] != true
            || replay["all_original_metrics_and_masks_replayed"] != true
            || replay["unseen_identification"] != false
        {
            return Err("complete original replay seal differs".into());
        }
        seconds += attempt.construction_seconds;
        if let Some(r) = &attempt.report {
            for e in &r.endpoints {
                if let Some(t) = &e.training {
                    gradients += t.work.gradient_batches;
                }
            }
        }
        for e in result["evaluation"]
            .as_array()
            .ok_or("missing evaluation outcomes")?
        {
            if e["status"] == "evaluated" {
                let reference = e["reference"]
                    .as_str()
                    .ok_or("missing actual model reference")?;
                if evaluated.insert(reference.to_string(), e.clone()).is_some() {
                    return Err("repeated constructed endpoint".into());
                }
            }
        }
        offers.extend(outcomes::offers(
            task,
            &attempt,
            &digest(&dir.join("attempt.json"))?,
        )?);
        records.push(json!({"task":task,"status":"complete","result":result,"replay":replay,
            "run_resources":resources,"replay_resources":parent::evidence(&root.join(format!("{task}.replay.resources")))?}));
    }
    let selected = comparison::select(&parent, &offers, &all)?;
    let arms = render(&parent, &selected, &evaluated)?;
    // Every staged construction is charged once, even if its later import or
    // evaluation fails. Original process evidence still measures import work.
    let staged = &inv["staging"]["summary"];
    if !inv["staging"].is_null() {
        seconds = staged["constructor_accounting"]["complete_constructor_seconds"]
            .as_f64()
            .ok_or("missing staged constructor time")?;
        gradients = staged["constructor_accounting"]["complete_gradient_batches"]
            .as_u64()
            .ok_or("missing staged derivative work")? as usize;
    }
    Ok(
        json!({"source_fingerprint":fingerprint(),"producer_commit":inv["producer_commit"],"protocol":protocol()?,
        "parent_sha256":inv["request"]["parent"]["summary_sha256"],"complete_parent":parent,
        "parent_snapshots":inv["parent_snapshots"],"parent_artifacts":inv["parent_artifacts"],
        "staged_construction":inv["staging"],
        "records":records,"tasks":all.len(),"process_failures":failures,"selection":selected,"arms":arms,
        "choose_between_initializers":false,"choices_per_arm":105,"all35_transformer_endpoints_retained":true,
        "unconditional_identification":false,"constructor_accounting":{"planned_cases":all.len(),
            "completed_cases":all.len()-failures,"complete_constructor_seconds":seconds,"complete_gradient_batches":gradients,
            "partial_costs_in_retained_resources":failures>0,"charged_once_across_both_arms":true,
            "construction_source":if inv["staging"].is_null(){"inline"}else{"sealed_training_stage"},
            "all_prior_costs_retained_in_complete_parent":true}}),
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
            .ok_or("missing initializer")?;
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
                let reference = r["selected"]
                    .as_str()
                    .ok_or("parent discarded without model")?;
                let new = evaluations
                    .get(reference)
                    .ok_or("selected actual model absent")?;
                if new["task"] != r["task"] || new["training_mode"] != r["training_mode"] {
                    return Err("selected model task/mode differs".into());
                }
                json!({"task":r["task"],"training_mode":r["training_mode"],"status":"selected","reference":reference,
                    "train_ce":new["train_ce"],"primary_split":new["primary_split"],"metrics":new["metrics"],
                    "primary_pass":new["primary_pass"],"all_splits_pass":new["all_splits_pass"],"saved_endpoint":new,"baseline":old})
            };
            choices.push(chosen);
        }
        let counts: Vec<_> = MODES.iter().map(|mode| {
            let rows: Vec<_> = choices.iter().filter(|c| c["training_mode"] == *mode).collect();
            json!({"training_mode":mode,"cases":rows.len(),"primary_passes":rows.iter().filter(|c| c["primary_pass"] == true).count(),
                "all_split_passes":rows.iter().filter(|c| c["all_splits_pass"] == true).count(),
                "no_model":rows.iter().filter(|c| c["status"] == "no_model").count()})
        }).collect();
        arms.push(
            json!({"initializer":arm["initializer"],"choices":choices,"by_mode":counts,
            "full35_pass":choices.iter().all(|c| c["all_splits_pass"] == true)}),
        );
    }
    Ok(arms)
}

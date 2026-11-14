//! Complete original scope, explicit metric provenance and all retained costs.
//! A failed case never acquires a success flag from its ancestor's metrics.
use super::*;
fn exit(root: &Path, file: &str) -> Result<i32> {
    let value = fs::read_to_string(root.join(file))?;
    let code = value.trim().parse::<i32>()?;
    if !(0..=255).contains(&code) {
        return Err("invalid process exit code".into());
    }
    Ok(code)
}
fn case(root: &Path, task: &str, inv: &Value) -> Result<Value> {
    let run = exit(root, &format!("{task}.run.exit"))?;
    let replay = if run == 0 {
        Some(exit(root, &format!("{task}.replay.exit"))?)
    } else {
        None
    };
    let mut artifacts = vec![];
    parents::tree(&root.join(task), &mut artifacts)?;
    for suffix in ["run.exit", "run.resources", "log"] {
        artifacts.push(parents::evidence(&root.join(format!("{task}.{suffix}")))?);
    }
    if replay.is_some() {
        for suffix in [
            "replay.exit",
            "replay.resources",
            "replay.json",
            "replay.log",
        ] {
            artifacts.push(parents::evidence(&root.join(format!("{task}.{suffix}")))?);
        }
    }
    let complete = run == 0 && replay == Some(0);
    let result = if complete {
        let value = execute::retained(root, task, inv)?;
        if read::<Value>(&root.join(format!("{task}.replay.json")))?
            != execute::replay_seal(root, task)?
        {
            return Err("changed case replay receipt differs".into());
        }
        Some(value)
    } else {
        None
    };
    Ok(
        json!({"task":task,"status":if complete{"complete"}else{"process_failure"},"run_exit":run,
      "replay_exit":replay,"artifacts":artifacts,"result":result}),
    )
}
pub(crate) fn render(
    parent: &Value,
    records: &[Value],
    all: &[String],
    parent_pin: &Reference,
) -> Result<Vec<Value>> {
    if records.len() != 35 || records.iter().zip(all).any(|(r, t)| r["task"] != *t) {
        return Err("complete ordered case universe required".into());
    }
    let mut arms = vec![];
    for policy in POLICIES {
        let mut choices = vec![];
        for (record, task) in records.iter().zip(all) {
            for mode in MODES {
                let old = cell(parent, policy, task, mode)?;
                let row = if record["status"] != "complete" {
                    json!({"task":task,"training_mode":mode,"status":"process_failure","reference":Value::Null,
                  "train_ce":Value::Null,"primary_pass":false,"all_splits_pass":false,"baseline":old,
                  "metric_provenance":{"kind":"unavailable_after_case_failure"}})
                } else {
                    let result = &record["result"];
                    let candidates = result["seal"]["choices"]["choices"]
                        .as_array()
                        .ok_or("structural choices omitted")?
                        .iter()
                        .filter(|c| {
                            c["initializer"] == policy
                                && c["task"] == *task
                                && c["training_mode"] == mode
                        })
                        .collect::<Vec<_>>();
                    if candidates.len() != 1 {
                        return Err("structural choice missing or repeated".into());
                    }
                    let selected = candidates[0];
                    if selected["prior_reference"] != old["reference"] {
                        return Err("original parent reference changed".into());
                    }
                    let outcomes = result["outcomes"]
                        .as_array()
                        .ok_or("changed model outcomes omitted")?
                        .iter()
                        .filter(|r| r["reference"] == selected["selected"])
                        .collect::<Vec<_>>();
                    if outcomes.len() > 1 {
                        return Err("changed model outcome repeated".into());
                    }
                    if let Some(new) = outcomes.first() {
                        if new["status"] != "evaluated"
                            || new["task"] != *task
                            || new["training_mode"] != mode
                            || new["train_ce"] != selected["train_ce"]
                            || new["metric_provenance"] != "fresh_changed_model"
                        {
                            return Err(
                                "fresh metric provenance differs from the sealed actual model"
                                    .into(),
                            );
                        }
                        json!({"task":task,"training_mode":mode,"status":"selected","reference":selected["selected"],
                      "train_ce":selected["train_ce"],"primary_split":new["primary_split"],"metrics":new["metrics"],
                      "primary_pass":new["primary_pass"],"all_splits_pass":new["all_splits_pass"],"saved_endpoint":new,
                      "baseline":old,"transformation":selected["transformation"],"metric_provenance":{"kind":"fresh_changed_model"}})
                    } else {
                        if selected["selected"] != old["reference"]
                            || selected["train_ce"] != old["train_ce"]
                        {
                            return Err("changed selected model has no fresh metric".into());
                        }
                        let mut inherited = old.clone();
                        inherited["metric_provenance"] = json!({"kind":"inherited_reviewed_parent","parent_summary_sha256":parent_pin.summary_sha256,
                      "parent_review_commit_sha256":parent_pin.review.commit_sha256,"reference":old["reference"]});
                        inherited["transformation"] = selected["transformation"].clone();
                        inherited
                    }
                };
                if row["status"] == "selected" {
                    let (primary, aux) = passes(
                        &row["metrics"],
                        row["primary_split"]
                            .as_str()
                            .ok_or("primary split omitted")?,
                    )?;
                    if row["primary_pass"] != primary || row["all_splits_pass"] != aux {
                        return Err("primary success differs from original rule".into());
                    }
                }
                choices.push(row);
            }
        }
        let by_mode=MODES.iter().map(|m|{
            let rows=choices.iter().filter(|r|r["training_mode"]==*m).collect::<Vec<_>>();
            json!({"training_mode":m,"cases":rows.len(),"primary_passes":rows.iter().filter(|r|r["primary_pass"]==true).count(),
              "all_split_passes":rows.iter().filter(|r|r["all_splits_pass"]==true).count(),
              "no_model":rows.iter().filter(|r|r["status"]!="selected").count(),
              "fresh_metrics":rows.iter().filter(|r|r["metric_provenance"]["kind"]=="fresh_changed_model").count(),
              "inherited_metrics":rows.iter().filter(|r|r["metric_provenance"]["kind"]=="inherited_reviewed_parent").count()})
        }).collect::<Vec<_>>();
        arms.push(
            json!({"initializer":policy,"choices":choices,"by_mode":by_mode,
          "full35_pass":choices.iter().all(|r|r["primary_pass"]==true),
          "all_split_diagnostic_pass":choices.iter().all(|r|r["all_splits_pass"]==true)}),
        );
    }
    Ok(arms)
}
pub fn summarize(root: &Path) -> Result<Value> {
    let (inv, request, reviewed) = execute::inventory(root, true)?;
    let all = tasks(&suite()?)?;
    let records = all
        .iter()
        .map(|t| case(root, t, &inv))
        .collect::<Result<Vec<_>>>()?;
    let parent = &reviewed["parent"].summary;
    let arms = render(parent, &records, &all, &request.parents["parent"])?;
    let changes = diagnostics::changes(parent, &arms)?;
    let complete = records.iter().filter(|r| r["status"] == "complete").count();
    let verification: f64 = records
        .iter()
        .filter_map(|r| r["result"]["verification_seconds"].as_f64())
        .sum();
    let metric: f64 = records
        .iter()
        .filter_map(|r| r["result"]["metric_seconds"].as_f64())
        .sum();
    if !verification.is_finite() || !metric.is_finite() {
        return Err("accumulated process times overflow".into());
    }
    Ok(
        json!({"source_fingerprint":fingerprint(),"binary_sha256":inv["binary_sha256"],"producer_commit":request.producer_commit,
      "request":request,"inventory_sha256":digest(&root.join("inventory.json"))?,"protocol":protocol()?,"tasks":35,
      "choices_per_arm":105,"choose_between_initializers":false,"all35_transformer_endpoints_retained":true,
      "success_rule":"original primary sequence accuracy >= 0.99; auxiliary splits are diagnostic",
      "arms":arms,"changes":changes,"records":records,"complete_parent":parent,
      "reviewed_sources":request.parents,"raw_training":reviewed["raw"].summary,"shared_training":reviewed["shared"].summary,
      "causal_training_already_charged_in_parent":true,"process_failures":35-complete,
      "constructor_accounting":{"charged_once_across_both_arms":true,"all_prior_costs_retained_in_complete_parent":true,
        "raw_training":reviewed["raw"].summary["constructor_accounting"],"shared_training":reviewed["shared"].summary["constructor_accounting"],
        "new_optimizer_updates":0,"new_training_gradient_batches":0,"verification_replays_saved_derivatives":true,
        "complete_case_verification_seconds":verification,"complete_case_metric_seconds":metric,"partial_costs_in_process_resources":complete!=35},
      "preparation_resources":parents::evidence(&root.join("prepare.resources"))?,"unconditional_identification":false,
      "raw_label_identification":false,"shared_role_target_proved":false,
      "metric_provenance":"fresh_changed_models_and_explicit_approved_parent_inheritance"}),
    )
}

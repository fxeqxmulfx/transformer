//! Complete ancestor scope and structural selection. No held-out scores
//! enter transform; primary/auxiliary outcomes are inspected only by render.
use super::*;
pub(crate) fn causal(parent: &Value) -> &Value {
    &parent["complete_parent"]["complete_parent"]["complete_parent"]
}
pub(crate) fn cell<'a>(
    parent: &'a Value,
    policy: &str,
    task: &str,
    mode: &str,
) -> Result<&'a Value> {
    let arms = parent["arms"]
        .as_array()
        .ok_or("missing initializer arms")?;
    let matches: Vec<_> = arms.iter().filter(|a| a["initializer"] == policy).collect();
    if matches.len() != 1 {
        return Err("unknown or repeated initializer".into());
    }
    let rows: Vec<_> = matches[0]["choices"]
        .as_array()
        .ok_or("missing choices")?
        .iter()
        .filter(|r| r["task"] == task && r["training_mode"] == mode)
        .collect();
    if rows.len() != 1 {
        return Err("unknown or repeated task/mode".into());
    }
    Ok(rows[0])
}
pub(crate) fn validate(parent: &Value, tasks: &[String]) -> Result<()> {
    let names: BTreeSet<_> = tasks.iter().map(String::as_str).collect();
    if names.len() != 35 || tasks.len() != 35 {
        return Err("complete35-task scope required".into());
    }
    let mut p = parent;
    for _ in 0..5 {
        if p["all35_transformer_endpoints_retained"] != true
            || p["choose_between_initializers"] != false
            || p["choices_per_arm"] != 105
            || p["constructor_accounting"]["charged_once_across_both_arms"] != true
            || p["constructor_accounting"]["all_prior_costs_retained_in_complete_parent"] != true
        {
            return Err("complete readout/kernel/refined/causal/quotient ancestry required".into());
        }
        let arms = p["arms"].as_array().ok_or("missing ancestor arms")?;
        if arms.len() != 2 {
            return Err("both initializer arms required".into());
        }
        for policy in POLICIES {
            let a: Vec<_> = arms.iter().filter(|a| a["initializer"] == policy).collect();
            if a.len() != 1 || a[0]["choices"].as_array().is_none_or(|v| v.len() != 105) {
                return Err("incomplete ancestor choices".into());
            }
            for task in tasks {
                for mode in MODES {
                    let c = cell(p, policy, task, mode)?;
                    match c["status"].as_str() {
                        Some("selected")
                            if c["reference"].is_string()
                                && c["train_ce"]
                                    .as_f64()
                                    .is_some_and(|v| v.is_finite() && v >= 0.) => {}
                        Some("no_model") if c["reference"].is_null() && c["train_ce"].is_null() => {
                        }
                        _ => return Err("invalid ancestor model/CE availability".into()),
                    }
                }
            }
        }
        p = &p["complete_parent"];
    }
    if p["depth_outcomes"]
        .as_array()
        .is_none_or(|v| v.len() != 336)
        || p["transformer_endpoints"]
            .as_array()
            .is_none_or(|v| v.len() != 35)
        || p["all_parent_costs_charged_once"] != true
    {
        return Err("complete depth/transformer scope and costs required".into());
    }
    Ok(())
}
pub(crate) fn transform(parent: &Value, task: &str, imports: &[Value]) -> Result<Value> {
    let mut rows = vec![];
    for policy in POLICIES {
        for m in MODES {
            let old = cell(parent, policy, task, m)?;
            let matching: Vec<_> = imports
                .iter()
                .filter(|r| r["initializer"] == policy && r["training_mode"] == m)
                .collect();
            if matching.len() != 1 {
                return Err("missing or repeated structural outcome".into());
            }
            let r = matching[0];
            let selected = if r["status"] == "wrapped" {
                r["reference"].clone()
            } else {
                old["reference"].clone()
            };
            if r["prior_reference"] != old["reference"]
                || (r["status"] == "wrapped" && r["train_ce"] != old["train_ce"])
            {
                return Err(
                    "structural transformation changed the selected training objective".into(),
                );
            }
            rows.push(json!({"initializer":policy,"task":task,"training_mode":m,"prior_reference":old["reference"],"selected":selected,"import_status":r["status"]}));
        }
    }
    Ok(
        json!({"choices":rows,"heldout_inputs":false,"observed_ce_head_selection_unchanged":true,"structural_transformation":"fixed_observed_support_parent_fallback"}),
    )
}
pub(crate) fn render(
    parent: &Value,
    evaluated: &BTreeMap<String, Value>,
    tasks: &[String],
) -> Result<Vec<Value>> {
    validate(parent, tasks)?;
    let mut arms = vec![];
    for policy in POLICIES {
        let mut choices = vec![];
        for task in tasks {
            for m in MODES {
                let old = cell(parent, policy, task, m)?;
                let key = format!("fallback:{policy}:{task}:{m}");
                let row = if let Some(new) = evaluated.get(&key) {
                    if new["status"] != "evaluated"
                        || new["prior_reference"] != old["reference"]
                        || new["train_ce"] != old["train_ce"]
                    {
                        return Err("evaluated wrapper is not the retained selected model".into());
                    }
                    json!({"task":task,"training_mode":m,"status":"selected","reference":key,"train_ce":new["train_ce"],"primary_split":new["primary_split"],"metrics":new["metrics"],"primary_pass":new["primary_pass"],"all_splits_pass":new["all_splits_pass"],"saved_endpoint":new,"baseline":old})
                } else {
                    old.clone()
                };
                if row["status"] == "selected" {
                    let (a, b) = passes(
                        &row["metrics"],
                        row["primary_split"]
                            .as_str()
                            .ok_or("missing primary split")?,
                    )?;
                    if row["primary_pass"] != a || row["all_splits_pass"] != b {
                        return Err("success differs from original metrics".into());
                    }
                }
                choices.push(row);
            }
        }
        let by_mode:Vec<_>=MODES.iter().map(|m|{let rows:Vec<_>=choices.iter().filter(|r|r["training_mode"]==*m).collect();json!({"training_mode":m,"cases":rows.len(),"primary_passes":rows.iter().filter(|r|r["primary_pass"]==true).count(),"all_split_passes":rows.iter().filter(|r|r["all_splits_pass"]==true).count(),"no_model":rows.iter().filter(|r|r["status"]=="no_model").count()})}).collect();
        arms.push(json!({"initializer":policy,"choices":choices,"by_mode":by_mode,"full35_pass":choices.iter().all(|r|r["primary_pass"]==true),"all_split_diagnostic_pass":choices.iter().all(|r|r["all_splits_pass"]==true)}));
    }
    Ok(arms)
}

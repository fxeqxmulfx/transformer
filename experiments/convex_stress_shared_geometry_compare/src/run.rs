//! Forecast and seal every endpoint before scoring. Reconstruct the actual old
//! states and repeat all new CE updates, then retain new scores unconditionally.
use super::*;
fn stage(root: &Path, entry: &Value, data: &data::Task) -> Result<model::Saved> {
    let sha = entry["source_model_sha256"]
        .as_str()
        .ok_or("missing source hash")?;
    let file = root.join(entry["source_file"].as_str().ok_or("missing source file")?);
    if digest(&file)? != sha {
        return Err("complete saved parent changed".into());
    }
    let parent: Value = read(&file)?;
    let mode = entry["training_mode"].as_str().ok_or("missing mode")?;
    let forecast = costs::forecast(
        &parent,
        sha,
        data,
        mode,
        fs::metadata(&file)?.len(),
        fs::metadata(root.join("baseline.json"))?.len(),
    )?;
    if forecast != entry["forecast"] {
        return Err("predeclared complete costs changed".into());
    }
    let inv = inputs::inventory(root)?;
    let request: Request = serde_json::from_value(inv["request"].clone())?;
    let make = |role: &str| -> Result<fresh::training::Outcome> {
        let input = convex_stress_shared_head::inspection::extract(&parent, role, sha)?;
        let policy = policy(mode)?;
        let key = format!("{:?}-{role}", policy.mode);
        let reference = request.component_root.join(&key).join("outcome.json");
        if digest(&reference)? != inv["protocol"]["outcomes"][&key] {
            return Err("reviewed component changed".into());
        }
        let outcome = fresh::training::run(input, policy)?;
        let prior: fresh::training::Outcome = read(&reference)?;
        if outcome != prior {
            return Err("fresh training differs from reviewed331 inspection".into());
        }
        Ok(outcome)
    };
    Ok(model::Saved {
        parent_sha256: sha.into(),
        prior: make("prior")?,
        supplement: if parent["supplement"].is_null() {
            None
        } else {
            Some(make("supplement")?)
        },
    })
}
fn execute(root: &Path, task: &str, replay: bool) -> Result<Value> {
    name(task)?;
    let inv = inputs::inventory(root)?;
    let cases: Vec<_> = inv["cases"]
        .as_array()
        .ok_or("missing cases")?
        .iter()
        .filter(|r| r["task"] == task)
        .collect();
    if cases.len() != 1 {
        return Err("original task missing or repeated".into());
    }
    let case = cases[0];
    let entries = case["models"].as_array().ok_or("missing selected models")?;
    let dir = root.join(task);
    if !replay {
        fs::create_dir(&dir)?;
        write(&dir.join("forecast.json"), &case)?;
    } else if read::<Value>(&dir.join("forecast.json"))? != *case {
        return Err("saved forecast changed".into());
    }
    let data = if entries.is_empty() {
        None
    } else {
        Some(inputs::dataset(task)?)
    };
    let mut seals = Vec::new();
    for (i, entry) in entries.iter().enumerate() {
        let model = stage(root, entry, data.as_ref().unwrap())?;
        let sha = fresh::digest(&model)?;
        let file = format!("model-{i}.json");
        if replay {
            if digest(&dir.join(&file))? != sha || read::<model::Saved>(&dir.join(&file))? != model
            {
                return Err("sealed state differs from fresh training".into());
            }
        } else {
            write(&dir.join(&file), &model)?;
        }
        seals.push(json!({"reference":entry["reference"],"training_mode":entry["training_mode"],"file":file,
            "model_sha256":sha,"parent_sha256":model.parent_sha256,"parent_file":entry["source_file"],"forecast":entry["forecast"]}));
    }
    let seal = json!({"task":task,"models":seals,"selection_before_metrics":true,"source_fingerprint":fingerprint(),
        "case_kernel_work_forecast":case["total_kernel_work"],"changed_endpoints":entries.len()});
    if replay {
        if read::<Value>(&dir.join("selection-seal.json"))? != seal {
            return Err("complete selection seal changed".into());
        }
    } else {
        write(&dir.join("selection-seal.json"), &seal)?;
    }
    let mut outcomes = Vec::new();
    for (i, (entry, sealed)) in entries.iter().zip(&seals).enumerate() {
        let saved_file = dir.join(sealed["file"].as_str().unwrap());
        let parent_file = root.join(entry["source_file"].as_str().unwrap());
        if digest(&saved_file)? != sealed["model_sha256"]
            || digest(&parent_file)? != entry["source_model_sha256"]
        {
            return Err("sealed complete state changed before score".into());
        }
        let model = model::Model::load(read(&parent_file)?, read(&saved_file)?)?;
        let data = data.as_ref().unwrap();
        let before = model::Predictor {
            model: &model,
            parent: true,
            calls: std::cell::Cell::new(0),
        };
        let after = model::Predictor {
            model: &model,
            parent: false,
            calls: std::cell::Cell::new(0),
        };
        let (mut scores, mut audits) = (serde_json::Map::new(), serde_json::Map::new());
        let mut audit_calls = 0u64;
        for (split, rows) in &data.splits {
            let previous = metrics::evaluate(&before, rows)?;
            if previous != entry["previous"]["metrics"][split] {
                return Err(format!("actual319 metrics differ on {task}/{split}").into());
            }
            let current = metrics::evaluate(&after, rows)?;
            let retained = inputs::retained_raw(root, task, i, split, replay)?;
            let audit = audit::run(
                &model,
                rows,
                &dir.join(format!("model-{i}-{}.jsonl", name(split)?)),
                replay,
                Some(&retained),
            )?;
            if correct(&current)? != audit["correct"]
                || correct(&previous)? != audit["parent_correct"]
                || current["targets"] != audit["targets"]
            {
                return Err("raw counts differ from independent full-logit audit".into());
            }
            for (metric, key) in [(&current, "new_ce"), (&previous, "old_ce")] {
                let a = metric["loss"].as_f64().ok_or("invalid CE")?;
                let b = audit[key].as_f64().ok_or("invalid audited CE")?;
                if (a - b).abs() > 1e-10 * (1. + a.abs()) {
                    return Err("ordinary CE audit differs".into());
                }
            }
            audit_calls += rows.len() as u64;
            scores.insert(split.clone(), current);
            audits.insert(split.clone(), audit);
        }
        let expected = &entry["forecast"]["calls"];
        if before.calls.get() != expected["old_metric_calls"]
            || after.calls.get() != expected["new_metric_calls"]
            || audit_calls != expected["old_raw_audit_calls"]
            || audit_calls != expected["new_raw_audit_calls"]
            || audit_calls != expected["independent_new_calls"]
        {
            return Err("actual call count differs from pre-score forecast".into());
        }
        let scores = Value::Object(scores);
        let primary = scores[&data.selection]["sequence_accuracy"]
            .as_f64()
            .ok_or("missing primary accuracy")?
            >= 0.99;
        let all = scores
            .as_object()
            .unwrap()
            .iter()
            .filter(|(s, _)| s.as_str() != "train")
            .all(|(_, m)| m["sequence_accuracy"].as_f64().is_some_and(|v| v >= 0.99));
        outcomes.push(json!({"task":task,"parent_reference":entry["reference"],"reference":format!("shared-geometry:{}",entry["reference"].as_str().unwrap()),
            "training_mode":entry["training_mode"],"primary_split":data.selection,"status":"selected",
            "metrics":scores,"parent_metrics":entry["previous"]["metrics"],"train_ce":scores["train"]["loss"],
            "primary_pass":primary,"all_splits_pass":all,"per_split_goals":goals::splits(&scores)?,
            "model_sha256":sealed["model_sha256"],"model_info":model.info(),"raw_transition_audit":audits,
            "actual_calls":expected,"forecast":entry["forecast"],"source_metrics_reproduced":true,
            "metric_provenance":{"kind":"fresh_changed_model"},"unconditional_identification":false}));
    }
    Ok(
        json!({"task":task,"status":"complete","eligible_models":entries.len(),"outcomes":outcomes,
        "selection_seal_sha256":digest(&dir.join("selection-seal.json"))?,"forecast_sha256":digest(&dir.join("forecast.json"))?,
        "unchanged_choice_provenance":if entries.is_empty(){"complete reviewed319 snapshot"}else{"new metrics from sealed shared-geometry states"}}),
    )
}
pub fn run_case(root: &Path, task: &str) -> Result<()> {
    let r = execute(root, task, false)?;
    write(&root.join(task).join("result.json"), &r)
}
pub fn replay_case(root: &Path, task: &str) -> Result<Value> {
    let r = execute(root, task, true)?;
    let path = root.join(task).join("result.json");
    if read::<Value>(&path)? != r {
        return Err("complete case replay changed".into());
    }
    Ok(
        json!({"task":task,"status":"replayed","result_sha256":digest(&path)?,"complete_raw_logits_reproduced":true}),
    )
}

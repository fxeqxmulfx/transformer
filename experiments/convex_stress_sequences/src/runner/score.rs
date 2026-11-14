use super::*;

pub fn measurements(stage: &Stage, data: &dataset::Dataset) -> Result<Value> {
    let mut modes = vec![];
    for (index, policy) in stage.request.training.iter().enumerate() {
        let mut error = stage.error.clone();
        let mut measurement = None;
        let mut work = Value::Null;
        if let Some(report) = &stage.model {
            error = report.initialization_error.clone();
            if error.is_none() {
                let endpoint = report.endpoints.get(index).ok_or("missing mode outcome")?;
                if endpoint.policy != *policy {
                    return Err("changed mode order".into());
                }
                error = endpoint.error.clone();
                if error.is_none() {
                    let training = endpoint.training.as_ref().ok_or("missing trained model")?;
                    work = json!({"training":training.work,"training_forecast":training.scalar_work_forecast,
                        "training_initial_ce":training.initial_ce,"training_final_ce":training.final_ce,
                        "audit":endpoint.audit});
                    match adapter::evaluate(&training.model, &data.task.splits) {
                        Ok(value) => measurement = Some(value),
                        Err(e) => error = Some(e.to_string()),
                    }
                }
            }
        }
        let mut primary = Value::Null;
        let mut test = Value::Null;
        let mut all_heldout99 = false;
        if let Some(measurement) = &measurement {
            primary = measurement["splits"][&data.task.selection]["goals"].clone();
            test = measurement["splits"]["test"]["goals"].clone();
            all_heldout99 = measurement["splits"]
                .as_object()
                .unwrap()
                .iter()
                .filter(|(name, _)| !name.starts_with("train"))
                .all(|(_, split)| split["goals"]["both99"] == true);
        }
        modes.push(json!({"mode":policy.mode,"policy":policy,"error":error,
            "measurement":measurement,"work":work,"primary":primary,"test":test,
            "all_heldout99":all_heldout99}));
    }
    Ok(json!({"case":data.name,"selection":data.task.selection,"modes":modes}))
}

fn historical(root: &Path, case: &str) -> Result<Value> {
    let history: Value = read(&root.join("history.json"))?;
    let rows: Vec<_> = history["runs"]
        .as_array()
        .ok_or("missing historical rows")?
        .iter()
        .filter(|r| r["case"] == case)
        .cloned()
        .collect();
    if rows.len() != 1 {
        return Err("historical case missing or duplicated".into());
    }
    Ok(
        json!({"snapshot_sha256":catalog()?["input_sha256"]["history"],
        "model_commit":catalog()?["historical_model_commit"],"row":rows[0],
        "same_historical_checkpoint_replayed":false,"historical_split_hash_available":false}),
    )
}

fn result(
    root: &Path,
    case: &str,
    data: &dataset::Dataset,
    stage: &Stage,
    seal: &Value,
) -> Result<Value> {
    Ok(
        json!({"training_seal":seal,"results":measurements(stage, data)?,
        "historical_gpt":historical(root, case)?}),
    )
}

pub fn score(root: &Path, case: &str) -> Result<()> {
    let (data, stage, seal) = load(root, case)?;
    write(
        &root.join(case).join("metrics.json"),
        &result(root, case, &data, &stage, &seal)?,
    )
}

pub fn replay(root: &Path, case: &str) -> Result<()> {
    let (data, stage, seal) = load(root, case)?;
    if let Some(report) = &stage.model {
        binding::replay(&stage.request, report)?;
    } else if construct(data.task.vocab, &data.task.splits["train"], data.metadata())? != stage {
        return Err("top-level binding refusal changed".into());
    }
    let saved: Value = read(&root.join(case).join("metrics.json"))?;
    let recomputed = result(root, case, &data, &stage, &seal)?;
    if !artifacts::same_json(&saved, &recomputed)? {
        return Err("complete metrics, complex prediction or refusal replay differs".into());
    }
    write(
        &root.join(case).join("replay.json"),
        &json!({"case":case,"exact_replay":true,
        "state_sha256":seal["state_sha256"],"metrics_sha256":digest(&root.join(case).join("metrics.json"))?}),
    )
}

pub fn summary(root: &Path) -> Result<Value> {
    let inventory = inventory(root)?;
    let mut rows = vec![];
    for case in cases()? {
        let (_, _, seal) = load(root, &case)?;
        let dir = root.join(&case);
        let metrics: Value = read(&dir.join("metrics.json"))?;
        let replay: Value = read(&dir.join("replay.json"))?;
        if metrics["training_seal"] != seal
            || metrics["historical_gpt"] != historical(root, &case)?
            || replay
                != json!({"case":case,"exact_replay":true,"state_sha256":seal["state_sha256"],
                "metrics_sha256":digest(&dir.join("metrics.json"))?})
        {
            return Err("missing or changed complete case replay".into());
        }
        rows.push(metrics);
    }
    let mut totals = vec![];
    for mode in ["Full", "Mini", "Forward"] {
        let endpoints: Vec<_> = rows
            .iter()
            .flat_map(|r| r["results"]["modes"].as_array().unwrap())
            .filter(|r| r["mode"] == mode)
            .collect();
        if endpoints.len() != 4 {
            return Err("omitted mode in summary denominator".into());
        }
        totals.push(json!({"mode":mode,"cases":4,
            "refusals":endpoints.iter().filter(|r| !r["error"].is_null()).count(),
            "original_successes":endpoints.iter().filter(|r| r["primary"]["original_success"] == true).count(),
            "primary_both99":endpoints.iter().filter(|r| r["primary"]["both99"] == true).count(),
            "test_both99":endpoints.iter().filter(|r| r["test"]["both99"] == true).count(),
            "all_heldout99":endpoints.iter().filter(|r| r["all_heldout99"] == true).count()}));
    }
    Ok(json!({"inventory":inventory,"cases":rows,"totals":totals,
        "complete":true,"new_cases":4,"remaining_additional_cases":126,
        "new_original35_evaluations":0,"entire165_evaluated":false}))
}

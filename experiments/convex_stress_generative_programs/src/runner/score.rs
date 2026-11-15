use super::*;
use crate::{evaluation, training};

pub fn measurements(stage: &Stage, data: &dataset::Dataset) -> Result<Value> {
    if let Some(report) = &stage.model
        && report.parent_usable()
    {
        return crate::parent::runner::measurements(
            &crate::parent::runner::Stage {
                dataset: stage.dataset.clone(),
                input: stage.input.clone(),
                model: report.parent.clone(),
                error: None,
            },
            data,
        );
    }
    let mut modes = vec![];
    for (index, policy) in training::policies()?.iter().enumerate() {
        let mut error = stage
            .error
            .clone()
            .or_else(|| stage.model.as_ref().and_then(|r| r.latent_error.clone()));
        let mut measurement = None;
        let mut work = Value::Null;
        if let Some(report) = stage.model.as_ref().and_then(|r| r.latent.as_ref()) {
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
                        "audit":report.audit});
                    match evaluation::evaluate(&training.model, &stage.input, &data.task.splits) {
                        Ok(value) => measurement = Some(value),
                        Err(e) => error = Some(e.to_string()),
                    }
                }
            }
        }
        let (mut primary, mut test) = (Value::Null, Value::Null);
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
            "measurement":measurement,"work":work,"primary":primary,"test":test,"all_heldout99":all_heldout99}));
    }
    Ok(json!({"case":data.name,"selection":data.task.selection,"modes":modes}))
}
fn result(
    root: &Path,
    case: &str,
    data: &dataset::Dataset,
    stage: &Stage,
    seal: &Value,
) -> Result<Value> {
    let prior = baseline_case(root, case)?;
    let measured = measurements(stage, data)?;
    let previous = crate::previous::case(root, case)?;
    crate::previous::check_case(
        &previous,
        &measured,
        seal["state_sha256"].as_str().ok_or("missing state hash")?,
    )?;
    let control = if crate::generative::CASES.contains(&case) {
        Value::Null
    } else {
        let report = stage.model.as_ref().ok_or("missing preserved control")?;
        if !report.parent_usable() || report.latent.is_some() || report.latent_error.is_some() {
            return Err("prior control model was replaced".into());
        }
        let old = crate::parent::runner::Stage {
            dataset: stage.dataset.clone(),
            input: stage.input.clone(),
            model: report.parent.clone(),
            error: None,
        };
        let mut bytes = serde_json::to_vec(&old)?;
        bytes.push(b'\n');
        let digest = hash(&bytes);
        if prior["training_seal"]["state_sha256"] != digest || prior["results"] != measured {
            return Err("complete prior control state/data/results differ".into());
        }
        json!({"state_sha256":digest,"state_data_results_exact":true})
    };
    Ok(
        json!({"training_seal":seal,"resource_accounting":crate::costs::report(stage,&measured)?,
        "results":measured,"control_preservation":control,
        "prior_baseline":prior,"previous_attempt":previous}),
    )
}
pub fn score(root: &Path, case: &str) -> Result<()> {
    all_sealed(root)?;
    let (data, stage, seal) = load(root, case)?;
    write(
        &root.join(case).join("metrics.json"),
        &result(root, case, &data, &stage, &seal)?,
    )
}
pub fn replay(root: &Path, case: &str) -> Result<()> {
    all_sealed(root)?;
    let (data, stage, seal) = load(root, case)?;
    if let Some(report) = &stage.model {
        binding::check(&stage.input, report)?;
    } else if construct(
        data.task.vocab,
        &data.task.splits["train"],
        dataset::metadata(&data),
    )? != stage
    {
        return Err("top-level shared-program refusal changed".into());
    }
    let saved: Value = read(&root.join(case).join("metrics.json"))?;
    if !artifacts::same_json(&saved, &result(root, case, &data, &stage, &seal)?)? {
        return Err("complete raw metric, prediction or refusal replay differs".into());
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
            || metrics["prior_baseline"] != baseline_case(root, &case)?
            || metrics["previous_attempt"] != crate::previous::case(root, &case)?
            || replay
                != json!({"case":case,"exact_replay":true,"state_sha256":seal["state_sha256"],
                "metrics_sha256":digest(&dir.join("metrics.json"))?})
        {
            return Err("missing or changed complete shared-program replay".into());
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
        if endpoints.len() != 12 {
            return Err("omitted mode in summary denominator".into());
        }
        totals.push(json!({"mode":mode,"cases":12,
            "refusals":endpoints.iter().filter(|r| !r["error"].is_null()).count(),
            "original_successes":endpoints.iter().filter(|r| r["primary"]["original_success"] == true).count(),
            "primary_both99":endpoints.iter().filter(|r| r["primary"]["both99"] == true).count(),
            "test_both99":endpoints.iter().filter(|r| r["test"]["both99"] == true).count(),
            "all_heldout99":endpoints.iter().filter(|r| r["all_heldout99"] == true).count()}));
    }
    let mut previous = crate::previous::read(root)?;
    previous
        .as_object_mut()
        .ok_or("invalid earlier report")?
        .remove("cases");
    Ok(
        json!({"inventory":inventory,"cases":rows,"totals":totals,"complete":true,
        "previous_attempt_summary":previous,"previous_attempt_cases_retained":12,
        "cohort_cases":12,"new_cases":6,"repeated_control_cases":6,
        "remaining_additional_cases":118,"new_original35_evaluations":0,
        "entire165_evaluated":false,"target_realizability_certified":false}),
    )
}

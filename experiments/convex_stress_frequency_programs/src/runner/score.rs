use super::*;

pub(super) fn measured(
    data: &frequency::dataset::Dataset,
    stage: &Stage,
    dir: &Path,
) -> Result<Value> {
    let mut endpoints = vec![];
    for (index, endpoint) in stage.endpoints.iter().enumerate() {
        let mut error = endpoint.error.clone();
        let mut measurement = None;
        if let Some(trained) = &endpoint.trained {
            match evaluation::evaluate(
                &trained.model,
                &stage.input,
                &data.task.splits,
                stage.metadata["final_token_diagnostic"] == true,
                &dir.join(format!("raw-{index}.jsonl")),
            ) {
                Ok(value) => measurement = Some(value),
                Err(e) => error = Some(e.to_string()),
            }
        }
        let primary = measurement.as_ref().map_or(Value::Null, |m| {
            m["splits"][&data.task.selection]["goals"].clone()
        });
        let test = measurement
            .as_ref()
            .map_or(Value::Null, |m| m["splits"]["test"]["goals"].clone());
        let all_heldout99 = measurement.as_ref().is_some_and(|m| {
            m["splits"]
                .as_object()
                .unwrap()
                .iter()
                .filter(|(name, _)| !name.starts_with("train"))
                .all(|(_, s)| s["goals"]["both99"] == true)
        });
        endpoints.push(json!({"prior":endpoint.prior,"mode":endpoint.policy.mode,"policy":endpoint.policy,
            "error":error,"primary":primary,"test":test,"all_heldout99":all_heldout99,"measurement":measurement,
            "training":endpoint.trained.as_ref().map(|t|json!({"initial_ce":t.training.initial_ce,
                "final_ce":t.training.final_ce,"work":t.training.work,"forecast":t.training.forecast,
                "complete_real_coefficients":t.model.head.coefficients.len(),"radius":t.model.head.radius,
                "actual_max_column_l1":t.model.head.max_column_l1(),"margin":t.model.head.margin,
                "selected_program":t.model.learned.program}))}));
    }
    Ok(
        json!({"case":data.name,"selection":data.task.selection,"endpoints":endpoints,
        "complete_case_forecast":stage.complete_case_forecast,"target_realizability_certified":false}),
    )
}

pub fn score(root: &Path, case: &str) -> Result<()> {
    let inv = inventory(root)?;
    all_sealed(root)?;
    let (data, stage, seal) = load(root, &inv, case)?;
    let dir = root.join(case);
    let results = measured(&data, &stage, &dir)?;
    write(
        &dir.join("metrics.json"),
        &json!({"seal":seal,"results":results}),
    )
}

pub fn replay(root: &Path, case: &str) -> Result<()> {
    let inv = inventory(root)?;
    all_sealed(root)?;
    let (data, stage, seal) = load(root, &inv, case)?;
    let dir = root.join(case);
    let replay_dir = dir.join("replay");
    fs::create_dir(&replay_dir)?;
    let results = measured(&data, &stage, &replay_dir)?;
    let replayed = json!({"seal":seal,"results":results});
    write(&replay_dir.join("metrics.json"), &replayed)?;
    let saved: Value = read(&dir.join("metrics.json"))?;
    if !artifacts::same_json(&saved, &replayed)? {
        return Err("complete metric/raw replay differs".into());
    }
    for index in 0..6 {
        let file = format!("raw-{index}.jsonl");
        let a = dir.join(&file);
        let b = replay_dir.join(&file);
        if a.exists() != b.exists() || (a.exists() && digest(&a)? != digest(&b)?) {
            return Err("complete raw logit files differ on fresh replay".into());
        }
    }
    write(
        &dir.join("replay.json"),
        &json!({"case":case,"exact_replay":true,
        "state_sha256":seal["state_sha256"],"metrics_sha256":digest(&dir.join("metrics.json"))?}),
    )
}

pub(super) fn totals(rows: &[Value]) -> Result<Vec<Value>> {
    let mut out = vec![];
    for prior in ["Coordinates", "Categorical"] {
        for mode in ["Forward", "Full", "Mini"] {
            let endpoints: Vec<_> = rows
                .iter()
                .flat_map(|r| r["results"]["endpoints"].as_array().into_iter().flatten())
                .filter(|e| e["prior"] == prior && e["mode"] == mode)
                .collect();
            if endpoints.len() != 7 {
                return Err("omitted prior or mode in seven-case denominator".into());
            }
            out.push(json!({"prior":prior,"mode":mode,"new_cases":7,
            "refusals":endpoints.iter().filter(|e|!e["error"].is_null()).count(),
            "original_successes":endpoints.iter().filter(|e|e["primary"]["original_success"]==true).count(),
            "primary_raw99":endpoints.iter().filter(|e|e["primary"]["raw99"]==true).count(),
            "primary_exact99":endpoints.iter().filter(|e|e["primary"]["exact99"]==true).count(),
            "primary_both99":endpoints.iter().filter(|e|e["primary"]["both99"]==true).count(),
            "test_both99":endpoints.iter().filter(|e|e["test"]["both99"]==true).count(),
            "all_heldout99":endpoints.iter().filter(|e|e["all_heldout99"]==true).count()}));
        }
    }
    Ok(out)
}

pub fn summary(root: &Path) -> Result<()> {
    let inv = inventory(root)?;
    all_sealed(root)?;
    let mut rows = vec![];
    for case in cases() {
        let dir = root.join(&case);
        let metric: Value = read(&dir.join("metrics.json"))?;
        let replay: Value = read(&dir.join("replay.json"))?;
        let seal: Value = read(&dir.join("seal.json"))?;
        if metric["seal"] != seal
            || replay
                != json!({"case":case,"exact_replay":true,
            "state_sha256":seal["state_sha256"],"metrics_sha256":digest(&dir.join("metrics.json"))?})
        {
            return Err("missing or changed complete replay".into());
        }
        for endpoint in metric["results"]["endpoints"]
            .as_array()
            .ok_or("missing endpoints")?
        {
            if let Some(m) = endpoint["measurement"].as_object() {
                let file = m["raw_file"].as_str().ok_or("missing complete raw file")?;
                if digest(&dir.join(file))? != m["raw_sha256"] {
                    return Err("raw vectors changed after replay".into());
                }
            }
        }
        rows.push(metric);
    }
    let totals = totals(&rows)?;
    let source: Value = read(&root.join("reference-data.json"))?;
    let original_sha = protocol()?["original35_summary_sha256"].clone();
    let fully_measured = rows
        .iter()
        .filter(|r| {
            let endpoints: Vec<_> = r["results"]["endpoints"]
                .as_array()
                .into_iter()
                .flatten()
                .filter(|e| e["prior"] == "Coordinates")
                .collect();
            endpoints.len() == 3
                && endpoints
                    .iter()
                    .all(|e| e["error"].is_null() && e["measurement"].is_object())
        })
        .count();
    let mut baseline_bytes = serde_json::to_vec(&source["baseline334"])?;
    baseline_bytes.push(b'\n');
    let baseline_hash = hash(&baseline_bytes);
    if baseline_hash != protocol()?["control334_report_sha256"] {
        return Err("complete inherited control report changed".into());
    }
    let mut cohort_totals = vec![];
    for new in &totals {
        let old = source["baseline334"]["totals"]
            .as_array()
            .ok_or("missing control totals")?
            .iter()
            .find(|old| old["mode"] == new["mode"])
            .ok_or("missing inherited mode")?;
        let mut combined = json!({"prior":new["prior"],"mode":new["mode"],"cases":23,"inherited_cases":16,"new_cases":7});
        for field in [
            "refusals",
            "original_successes",
            "primary_both99",
            "test_both99",
            "all_heldout99",
        ] {
            combined[field] = json!(
                old[field].as_u64().ok_or("invalid inherited count")?
                    + new[field].as_u64().ok_or("invalid new count")?
            );
        }
        cohort_totals.push(combined);
    }
    let report = json!({"inventory":inv,"cases":rows,"totals":totals,"cohort_totals":cohort_totals,"complete":true,
        "cohort_cases":23,"new_cases":7,"inherited_control_cases":16,"baseline334":source["baseline334"],
        "fully_measured_new_cases":fully_measured,"control334_exact_sha256":baseline_hash,
        "historical_frequency_cases":source["cases"],"remaining_additional_cases":114-fully_measured,
        "original35_reference":{"file":"original35.json","sha256":original_sha,"new_evaluations":0},
        "entire165_evaluated":false,"target_realizability_certified":false});
    write(&root.join("report.json"), &report)
}

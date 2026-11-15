use super::*;

#[test]
fn previous_training_serialization_and_forecast_only_change_are_checked() -> Result<()> {
    let prepared = Prepared {
        metadata: json!({"original_rows":7}),
        training: None,
        complete_forecast: json!({"new_numerical_cost":123}),
        error: Some("original refusal".into()),
    };
    let old_forecast = json!({"original_cost":100});
    let original = previous::runner::Prepared {
        metadata: prepared.metadata.clone(),
        training: prepared.training.clone(),
        complete_forecast: old_forecast.clone(),
        error: prepared.error.clone(),
    };
    let hash = artifacts::canonical_digest(&original)?;
    assert_eq!(original_state_digest(&prepared, &old_forecast)?, hash);
    let root = std::env::temp_dir().join(format!("cert349-bind-{}", std::process::id()));
    fs::create_dir(&root)?;
    write(
        &root.join("baseline347.json"),
        &json!({"cases":[{
        "results":{"case":"fixture","complete_case_forecast":old_forecast},
        "seal":{"state_sha256":hash}}]}),
    )?;
    assert_eq!(unchanged_training(&root, "fixture", &prepared)?, hash);
    let mut altered = prepared.clone();
    altered.metadata["original_rows"] = json!(6);
    assert!(unchanged_training(&root, "fixture", &altered).is_err());
    altered = prepared.clone();
    altered.error = None;
    assert!(unchanged_training(&root, "fixture", &altered).is_err());
    assert!(unchanged_training(&root, "missing", &prepared).is_err());
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn numerical_protocol_keeps_all_discovery_and_training_settings_fixed() -> Result<()> {
    let new = protocol()?;
    let old: Value = serde_json::from_str(previous::PROTOCOL)?;
    for field in [
        "fit",
        "selector",
        "priors",
        "categorical_control",
        "initial_mass_scaling",
        "original_success",
        "goals",
        "process_address_space_bytes",
        "buffer_ceiling_bytes",
    ] {
        assert_eq!(new[field], old[field], "changed {field}");
    }
    Ok(())
}

#[test]
fn raw_only_success_does_not_receive_sequence_or_refused_endpoint_credit() -> Result<()> {
    let mut rows: Vec<_> = cases()
        .iter()
        .enumerate()
        .map(|(i, case)| {
            let endpoints: Vec<_> = ["Coordinates", "Categorical"]
                .into_iter()
                .flat_map(|prior| {
                    ["Forward", "Full", "Mini"].into_iter().map(move |mode| {
                        json!({
                "prior":prior,"mode":mode,"error":null,"measurement":{},
                "all_heldout99":i<11,"all_heldout_exact99":i<11,"all_heldout_raw99":i<12})
                    })
                })
                .collect();
            json!({"results":{"case":case,"endpoints":endpoints}})
        })
        .collect();
    assert_eq!(summary::passing_cases(&rows)?.len(), 11);
    assert_eq!(summary::raw_passing_cases(&rows)?.len(), 12);
    rows[11]["results"]["endpoints"][0]["error"] = json!("numerical refusal");
    assert_eq!(summary::raw_passing_cases(&rows)?.len(), 11);
    Ok(())
}

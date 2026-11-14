use super::*;

fn inputs() -> Result<(String, Value, Value, Value)> {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let reference = std::env::var("STRESS_REFERENCE_ROOT")?;
    let reference = Path::new(&reference).join("research/current_grokking");
    Ok((
        checked_text("catalog", &repo.join("stress_tests_for_gpt.md"))?,
        serde_json::from_str(&checked_text("plan", &reference.join("plan.json"))?)?,
        serde_json::from_str(&checked_text(
            "existing35",
            &repo.join("experiments/convex_stress/suite.json"),
        )?)?,
        serde_json::from_str(&checked_text(
            "history",
            &reference.join("results/progress_snapshot.json"),
        )?)?,
    ))
}

#[test]
fn exact_sources_keep_all_cases_original_targets_and_new_goals_separate() -> Result<()> {
    let (catalog, plan, existing, history) = inputs()?;
    let report = inventory(&catalog, &plan, &existing, &history)?;
    let cases = report["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 165);
    assert_eq!(
        cases
            .iter()
            .filter(|r| r["additional_case"] == true)
            .count(),
        130
    );
    assert_eq!(
        cases.iter().filter(|r| r["first_cohort"] == true).count(),
        4
    );
    for row in cases.iter().filter(|r| r["first_cohort"] == true) {
        assert_eq!(row["original_case"]["target"], 0.95);
        assert_eq!(row["new_accuracy_targets"]["raw_token"], 0.99);
        assert_eq!(row["new_accuracy_targets"]["exact_sequence"], 0.99);
    }
    assert!(cases.iter().all(|r| r["expanded_model_result"].is_null()));
    assert_eq!(report["historical_retained_targets"], 40);
    Ok(())
}

#[test]
fn missing_duplicate_or_mismatched_cases_are_refused() -> Result<()> {
    let (catalog, plan, existing, history) = inputs()?;
    let mut missing = plan.clone();
    missing["cases"].as_array_mut().unwrap().pop();
    assert!(inventory(&catalog, &missing, &existing, &history).is_err());
    let mut duplicate = plan.clone();
    duplicate["cases"][1] = duplicate["cases"][0].clone();
    assert!(inventory(&catalog, &duplicate, &existing, &history).is_err());
    let mut target = plan.clone();
    target["cases"][0]["target"] = json!(0.95);
    assert!(inventory(&catalog, &target, &existing, &history).is_err());
    let mut bad_history = history.clone();
    bad_history["runs"][0]["status"] = json!("pending");
    assert!(inventory(&catalog, &plan, &existing, &bad_history).is_err());
    let mut bad_existing = existing.clone();
    bad_existing["tasks"][0]["name"] = json!("invented-case");
    assert!(inventory(&catalog, &plan, &bad_existing, &history).is_err());
    let duplicate_catalog = format!("{catalog}\n| `suite-mqar` | Extra | budget | — |\n");
    assert!(inventory(&duplicate_catalog, &plan, &existing, &history).is_err());
    Ok(())
}

#[test]
fn wrong_source_identity_is_refused_before_inventory() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("protocol.json");
    assert!(checked_text("plan", &path).is_err());
    assert!(checked_text("unregistered-source", &path).is_err());
}

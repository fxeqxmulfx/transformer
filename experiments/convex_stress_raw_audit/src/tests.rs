use super::*;
fn summary() -> Value {
    let records: Vec<_> = (0..35)
        .map(|i| json!({"task":format!("task{i}"),"status":"complete"}))
        .collect();
    let choices:Vec<_>=(0..35).flat_map(|i|["full_batch","minibatch","forward_only"].map(move |mode|json!({
        "task":format!("task{i}"),"training_mode":mode,"primary_split":"validation","reference":"fixture",
        "metrics":{"validation":{"targets":100,"native_correct":90,"native_token_accuracy":0.9,"correct":100,"token_accuracy":1.}}}))).collect();
    json!({"tasks":35,"choices_per_arm":105,"process_failures":0,"records":records,
        "arms":[{"initializer":"minimum_ce","choices":choices},{"initializer":"first_small_ce","choices":choices}]})
}
#[test]
fn raw_totals_use_native_counts_and_both_complete_policies() -> Result<()> {
    let result = counts::summarize(&summary())?;
    for arm in result.as_array().unwrap() {
        for mode in arm["modes"].as_array().unwrap() {
            assert_eq!(mode["targets"], 3500);
            assert_eq!(mode["correct"], 3150);
            assert_eq!(mode["errors"], 350);
            assert_eq!(mode["accuracy"], 0.9);
        }
    }
    Ok(())
}
#[test]
fn incomplete_scope_duplicates_and_mismatched_raw_counts_are_rejected() {
    for variant in 0..5 {
        let mut bad = summary();
        match variant {
            0 => {
                bad["records"].as_array_mut().unwrap().pop();
            }
            1 => {
                bad["arms"][0]["choices"][0] = bad["arms"][0]["choices"][1].clone();
            }
            2 => {
                bad["arms"][0]["choices"][0]["metrics"]["validation"]["native_correct"] = json!(91);
            }
            3 => {
                bad["arms"][0]["choices"][0]["metrics"]["validation"]["targets"] = json!(0);
            }
            _ => {
                bad["arms"][1] = bad["arms"][0].clone();
            }
        }
        assert!(counts::summarize(&bad).is_err());
    }
}
#[test]
fn affine_diagnosis_distinguishes_extrapolation_from_an_unverified_region() -> Result<()> {
    let points = [((0, 0), 0), ((2, 0), 0), ((0, 1), 2), ((2, 1), 2)];
    let line = geometry::describe(&points, 0, (4, 0))?;
    assert_eq!(line["affine_rank"], 1);
    assert_eq!(line["query_in_affine_span"], true);
    assert_eq!(
        line["line_intersects_other_observed_class_hulls"],
        json!([])
    );
    assert_eq!(
        geometry::describe(&points, 0, (4, 1))?["query_in_affine_span"],
        false
    );
    let crossing = [((0, 0), 0), ((2, 0), 0), ((0, 1), 2), ((2, -1), 2)];
    assert_eq!(
        geometry::describe(&crossing, 0, (4, 0))?["line_intersects_other_observed_class_hulls"],
        json!([2])
    );
    Ok(())
}
#[test]
fn empty_singleton_and_area_classes_have_distinct_ranks() -> Result<()> {
    let points = [((0, 0), 0), ((2, 0), 0), ((0, 2), 0), ((1, 1), 2)];
    assert_eq!(geometry::describe(&points, 0, (9, 9))?["affine_rank"], 2);
    assert_eq!(geometry::describe(&points, 2, (9, 9))?["affine_rank"], 0);
    assert_eq!(
        geometry::describe(&points, 2, (9, 9))?["query_in_affine_span"],
        false
    );
    assert_eq!(
        geometry::describe(&points, 3, (9, 9))?["observed_points"],
        0
    );
    Ok(())
}
#[test]
fn diagnostic_geometry_refuses_integer_overflow() {
    let points = [
        ((i64::MIN, i64::MIN), 0),
        ((i64::MAX, i64::MIN), 0),
        ((i64::MIN, i64::MAX), 0),
    ];
    assert!(geometry::describe(&points, 0, (0, 0)).is_err());
}
#[test]
fn argmax_matches_original_first_index_and_signed_zero_convention() -> Result<()> {
    assert_eq!(inspect::argmax(&[1., 1., 0.])?, 0);
    assert_eq!(inspect::argmax(&[-0., 0.])?, 1);
    assert!(inspect::argmax(&[]).is_err());
    assert!(inspect::argmax(&[f64::NAN]).is_err());
    Ok(())
}
#[test]
fn artifact_mutations_and_absent_files_cannot_pass_pins() -> Result<()> {
    let dir = std::env::temp_dir().join(format!("raw-audit-pins-{}", std::process::id()));
    fs::create_dir(&dir)?;
    let path = dir.join("source.json");
    fs::write(&path, b"original")?;
    let manifest = BTreeMap::from([(path.to_str().unwrap().into(), digest(&path)?)]);
    pinned(&path, &manifest)?;
    fs::write(&path, b"changed")?;
    assert!(pinned(&path, &manifest).is_err());
    assert!(pinned(&dir.join("missing"), &manifest).is_err());
    fs::remove_dir_all(dir)?;
    Ok(())
}
#[test]
fn compiled_parent_matches_the_exact_reviewed_comparison() -> Result<()> {
    let p: Value = serde_json::from_str(PROTOCOL)?;
    assert_eq!(
        convex_stress_raw_regions_compare::fingerprint(),
        p["source_fingerprint"]
    );
    assert_eq!(p["prediction_work_limit_per_endpoint"], 128_000_000_000_u64);
    Ok(())
}

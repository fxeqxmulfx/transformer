//! Ported from task306 at65aa7961; native counts take precedence over field scores.
use super::*;
fn fixture() -> Value {
    let records: Vec<_> = (0..35)
        .map(|i| json!({"task":format!("task{i}"),"status":"complete"}))
        .collect();
    let choices:Vec<_>=(0..35).flat_map(|i|["full_batch","minibatch","forward_only"].map(move|mode|json!({
        "task":format!("task{i}"),"training_mode":mode,"primary_split":"validation","reference":"fixture",
        "metrics":{"validation":{"targets":100,"native_correct":90,"native_token_accuracy":0.9,"correct":100,"token_accuracy":1.}}}))).collect();
    json!({"tasks":35,"choices_per_arm":105,"process_failures":0,"records":records,
        "arms":[{"initializer":"minimum_ce","choices":choices},{"initializer":"first_small_ce","choices":choices}]})
}
#[test]
fn native_counts_and_both_complete_policies_determine_raw_totals() -> Result<()> {
    for arm in counts::summarize(&fixture())?.as_array().unwrap() {
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
fn raw_totals_reject_missing_scope_duplicates_and_inconsistent_counts() {
    for variant in 0..5 {
        let mut bad = fixture();
        match variant {
            0 => {
                bad["records"].as_array_mut().unwrap().pop();
            }
            1 => bad["arms"][0]["choices"][0] = bad["arms"][0]["choices"][1].clone(),
            2 => {
                bad["arms"][0]["choices"][0]["metrics"]["validation"]["native_correct"] = json!(91)
            }
            3 => bad["arms"][0]["choices"][0]["metrics"]["validation"]["targets"] = json!(0),
            _ => bad["arms"][1] = bad["arms"][0].clone(),
        }
        assert!(counts::summarize(&bad).is_err());
    }
}
#[test]
fn raw_audit_argmax_matches_original_tie_convention() -> Result<()> {
    assert_eq!(model::argmax(&[1., 1., 0.])?, 0);
    assert_eq!(model::argmax(&[-0., 0.])?, 1);
    assert!(model::argmax(&[]).is_err());
    assert!(model::argmax(&[f64::NAN]).is_err());
    Ok(())
}

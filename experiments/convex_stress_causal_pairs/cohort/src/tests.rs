//! Synthetic original-layout evidence; no benchmark task or answer is read.
use super::*;
mod boundaries;
mod fixtures;
mod staged;
mod validation;
use fixtures::*;
#[test]
fn actual_pair_head_uses_original_generation_and_full_vocabulary_ce() -> Result<()> {
    let f = Fixture::new()?;
    let (inv, parent) = f.prepare()?;
    f.successful(0, &inv, &parent)?;
    let dir = f.run.join("fixture-0");
    let attempt: execute::Attempt = read(&dir.join("attempt.json"))?;
    let report = attempt.report.unwrap();
    assert!(matches!(
        report
            .discovery
            .groups
            .iter()
            .find(|g| g.query == 6)
            .unwrap()
            .decoder
            .as_ref()
            .unwrap()
            .candidate,
        component::features::Candidate::Pair(_, _)
    ));
    let result: Value = read(&dir.join("result.json"))?;
    for row in result["evaluation"].as_array().unwrap() {
        assert_eq!(row["all_splits_pass"], true);
        assert_eq!(row["metrics"]["test"]["eos_rate"], 1.0);
        assert_eq!(row["metrics"]["test"]["generated_tokens"], 8);
        assert_eq!(row["unknown_prefixes"]["test"]["calls"], 16);
        assert_eq!(row["unknown_prefixes"]["test"]["unknown_calls"], 0);
        assert_eq!(row["audit"]["unseen_identification"], false);
    }
    assert_eq!(
        report.endpoints[2]
            .training
            .as_ref()
            .unwrap()
            .work
            .gradient_examples,
        0
    );
    assert_eq!(report.initialization.as_ref().unwrap().ce_gradients, 0);
    assert!(result["selection"]["heldout_inputs"] == false);
    Ok(())
}
#[test]
fn complete_parent_and_every_process_failure_remain_in_all35_comparison() -> Result<()> {
    let f = Fixture::new()?;
    let (inv, parent) = f.prepare()?;
    f.successful(0, &inv, &parent)?;
    f.successful(2, &inv, &parent)?; // Explicit two-token shape refusal, no fitting.
    for i in (1..35).filter(|&i| i != 2) {
        f.failed(i)?;
    }
    let summary = summary::summarize_with(&f.run, &inv, parent.clone())?;
    assert_eq!(summary["complete_parent"], parent);
    assert_eq!(summary["records"].as_array().unwrap().len(), 35);
    assert_eq!(summary["process_failures"], 33);
    assert_eq!(summary["all35_transformer_endpoints_retained"], true);
    assert_eq!(
        summary["constructor_accounting"]["complete_gradient_batches"],
        272
    );
    for arm in summary["arms"].as_array().unwrap() {
        assert_eq!(arm["choices"].as_array().unwrap().len(), 105);
        assert_eq!(arm["by_mode"][2]["no_model"], 1);
        assert_eq!(arm["by_mode"][0]["cases"], 35);
        assert_eq!(arm["full35_pass"], false);
    }
    let row = &summary["records"][1];
    assert_eq!(row["prior_choices_retained"], 6);
    assert_eq!(row["partial_artifacts"].as_array().unwrap().len(), 1);
    Ok(())
}
#[test]
fn input_shape_and_selection_are_independent_of_heldout_answers_and_names() -> Result<()> {
    let mut t = task(false);
    let original = inputs::assemble(task(false), "fixture-0", json!({}))?;
    t.selection = "other".into();
    t.splits.insert("other".into(), t.splits["test"].clone());
    for e in t.splits.get_mut("test").unwrap() {
        e.targets.fill(0);
        e.tokens.resize(99, 1);
    }
    let changed = inputs::assemble(t, "arbitrary-name", json!({}))?;
    assert_eq!(original.configuration, changed.configuration);
    let skipped = inputs::assemble(task(true), "not-an-arithmetic-name", json!({}))?;
    assert!(skipped.configuration.is_none());
    assert_eq!(skipped.record["shape_eligible"], false);
    assert!(skipped.refusal.unwrap().contains("two tokens"));
    let (parent, names) = parent_value()?;
    let offer = comparison::Offer {
        task: names[0].clone(),
        mode: training::Mode::Full,
        train_ce: 0.1,
        model_sha256: "a".repeat(64),
        report_sha256: "b".repeat(64),
    };
    let selected = comparison::select(&parent, std::slice::from_ref(&offer), &names)?;
    let mut changed = parent;
    for arm in changed["arms"].as_array_mut().unwrap() {
        for row in arm["choices"].as_array_mut().unwrap() {
            row["metrics"] = json!({"test":{"sequence_accuracy":0.0}});
        }
    }
    assert_eq!(selected, comparison::select(&changed, &[offer], &names)?);
    Ok(())
}

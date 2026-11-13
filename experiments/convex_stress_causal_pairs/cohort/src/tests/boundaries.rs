use super::*;
fn unknown_input() -> Result<inputs::Inputs> {
    let mut x = task(false);
    let e = &mut x.splits.get_mut("test").unwrap()[0];
    e.tokens[0] = 2;
    e.prompt[0] = 2;
    inputs::assemble(x, "fixture-0", json!({"synthetic":true}))
}
#[test]
fn unknown_actual_greedy_prefixes_remain_visible_and_never_become_certificates() -> Result<()> {
    let f = Fixture::new()?;
    let (inv, parent) = f.prepare()?;
    execute::run_with(&f.run, "fixture-0", &inv, &parent, unknown_input()?)?;
    execute::replay_with(&f.run, "fixture-0", &inv, &parent, unknown_input()?)?;
    let result: Value = read(&f.run.join("fixture-0/result.json"))?;
    for row in result["evaluation"].as_array().unwrap() {
        assert_eq!(row["metrics"]["train"]["sequence_accuracy"], 1.0);
        assert_eq!(row["metrics"]["test"]["sequence_accuracy"], 0.75);
        assert_eq!(row["metrics"]["test"]["eos_rate"], 0.75);
        assert!(
            row["unknown_prefixes"]["test"]["unknown_calls"]
                .as_u64()
                .unwrap()
                > 0
        );
        assert_eq!(row["all_splits_pass"], false);
        assert_eq!(row["unseen_identification"], false);
    }
    Ok(())
}
#[test]
fn original_prefix_caps_are_recorded_before_expanding_training_storage() -> Result<()> {
    let mut x = task(false);
    let e = &mut x.splits.get_mut("train").unwrap()[0];
    e.tokens = vec![1; 4097];
    e.targets = vec![-100; 4097];
    e.targets[4096] = 3;
    e.changes = vec![false; 4097];
    let input = inputs::assemble(x, "fixture-0", json!({"synthetic":true}))?;
    assert!(input.configuration.is_none());
    assert_eq!(input.record["training_context"], 4097);
    assert_eq!(input.record["shape_eligible"], true);
    assert!(input.refusal.unwrap().contains("input caps"));
    Ok(())
}

use super::{fixtures::*, *};
#[test]
fn original_inputs_use_categorical_pairs_raw_targets_and_only_declared_readout() -> Result<()> {
    let x = input("arbitrary-name", false)?;
    assert_eq!(x.config.input.n, 7);
    assert_eq!(x.config.input.vocabulary, 7);
    for (e, row) in x.task.splits["train"]
        .iter()
        .zip(&x.config.input.observations)
    {
        assert_eq!(
            *row,
            [
                e.tokens[1] as usize + 7 * e.tokens[0] as usize,
                e.targets[1] as usize
            ]
        );
    }
    let mut changed = task(false);
    for e in changed.splits.get_mut("validation").unwrap() {
        e.targets[1] = 0;
    }
    assert_eq!(
        x.config,
        inputs::assemble(changed, "renamed", json!({}))?.config
    );
    let mut field = task(false);
    field.vocab = 200;
    for e in field.splits.values_mut().flatten() {
        e.field = Some((0, 0));
    }
    let first = inputs::assemble(field, "field", json!({}))?;
    assert_eq!(
        first.config.readout.as_ref().unwrap()[1],
        first.config.readout.as_ref().unwrap()[98]
    );
    let mut changed = first.task;
    for e in changed.splits.values_mut().flatten() {
        e.field = Some((73, 92));
    }
    assert_eq!(
        first.config,
        inputs::assemble(changed, "different-name", json!({}))?.config
    );
    let mut mixed = task(false);
    mixed.splits.get_mut("train").unwrap()[0].field = Some((1, 2));
    assert!(inputs::assemble(mixed, "mixed", json!({})).is_err());
    assert!(inputs::assemble(task(false), "../unsafe", json!({})).is_err());
    Ok(())
}
#[test]
fn complete_original_fixture_preserves_both_arms_transformer_failures_and_costs() -> Result<()> {
    let f = Fixture::new()?;
    let (inv, parent) = f.prepare(false)?;
    f.successful(&inv, &parent, false)?;
    f.failed()?;
    let out = summary::summarize_with(&f.destination, &inv, parent.clone(), &f.planned)?;
    assert_eq!(out["complete_parent"], parent);
    assert_eq!(out["process_failures"], 1);
    assert_eq!(
        out["constructor_accounting"]["complete_stage_constructions"],
        55
    );
    assert_eq!(
        out["constructor_accounting"]["complete_gradient_batches"],
        272
    );
    assert_eq!(
        out["records"][1]["partial_artifacts"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert_eq!(out["records"][1]["run_exit"], 137);
    assert_eq!(out["records"][0]["result"]["eligible_offers"], 3);
    for arm in 0..2 {
        assert_eq!(out["arms"][arm]["choices"].as_array().unwrap().len(), 105);
        for i in 0..3 {
            assert_eq!(out["arms"][arm]["choices"][i]["all_splits_pass"], true);
        }
        for i in 3..105 {
            assert_eq!(
                out["arms"][arm]["choices"][i],
                parent["arms"][arm]["choices"][i]
            );
        }
    }
    assert_eq!(out["records"][0]["replay"]["all_ce_steps_replayed"], true);
    assert_eq!(out["unconditional_identification"], false);
    assert_eq!(
        out["complete_parent"]["transformer_endpoints"]
            .as_array()
            .unwrap()
            .len(),
        35
    );
    Ok(())
}
#[test]
fn incompatible_alias_codes_are_evaluated_but_never_offered_as_raw_identification() -> Result<()> {
    let f = Fixture::new()?;
    let (inv, parent) = f.prepare(true)?;
    f.successful(&inv, &parent, true)?;
    f.failed()?;
    let out = summary::summarize_with(&f.destination, &inv, parent.clone(), &f.planned)?;
    let result = &out["records"][0]["result"];
    assert_eq!(result["eligible_offers"], 0);
    assert_eq!(result["evaluation"].as_array().unwrap().len(), 6);
    assert!(
        result["evaluation"]
            .as_array()
            .unwrap()
            .iter()
            .all(|e| e["offer_eligible"] == false && e["raw_label_identification"] == false)
    );
    for arm in 0..2 {
        assert_eq!(out["arms"][arm]["choices"], parent["arms"][arm]["choices"]);
    }
    Ok(())
}
#[test]
fn complete_parent_model_review_and_local_snapshot_changes_are_rejected() -> Result<()> {
    let f = Fixture::new()?;
    let (inv, _) = f.prepare(false)?;
    for path in [
        f.request.parent.root.join("summary.json"),
        f.request.parent.root.join("ancestor-0.json"),
        f.temp.0.join("ancestor-0.json"),
        f.temp.0.join("old-model.json"),
        f.request.parent.binary.clone(),
        f.request.parent.review.replay.clone(),
        f.request.parent.review.exit.clone(),
        f.request.parent.review.commit.clone(),
    ] {
        let saved = fs::read(&path)?;
        fs::write(&path, b"changed")?;
        assert!(
            inventory::inventory_with(&f.destination, &f.suite, &f.planned, true).is_err(),
            "{}",
            path.display()
        );
        fs::write(&path, saved)?;
    }
    let path = f.destination.join("parent/ancestor-1.json");
    fs::write(&path, b"changed")?;
    assert!(
        parent::check_snapshot(&f.destination.join("parent"), &inv["parent_snapshots"]).is_err()
    );
    Ok(())
}
#[test]
fn changed_metrics_selection_inputs_or_model_are_rejected_by_original_replay() -> Result<()> {
    let f = Fixture::new()?;
    let (inv, parent) = f.prepare(false)?;
    f.successful(&inv, &parent, false)?;
    let name = &f.planned[0];
    let dir = f.destination.join(name);
    let file = dir.join("result.json");
    let saved: Value = read(&file)?;
    for mutation in 0..5 {
        let mut bad = saved.clone();
        match mutation {
            0 => bad["evaluation"][0]["metrics"]["validation"]["sequence_accuracy"] = json!(0.),
            1 => {
                bad["selection"]["selection"]["arms"][0]["choices"][0]["selected"] = json!("forged")
            }
            2 => bad["eligible_offers"] = json!(0),
            3 => bad["metadata"]["input"]["primary_split"] = json!("train"),
            _ => bad["evaluation"][0]["train_ce"] = json!(0.),
        }
        overwrite(&file, &bad);
        assert!(
            execute::replay_with(&f.destination, name, &inv, &parent, input(name, false)?).is_err(),
            "mutation{mutation}"
        );
    }
    overwrite(&file, &saved);
    let mut x = input(name, false)?;
    x.config.input.observations[0][1] = 0;
    assert!(execute::replay_with(&f.destination, name, &inv, &parent, x).is_err());
    let root = dir.join("catalogue");
    let mut ledger: Ledger = read(&root.join("ledger.json"))?;
    let slot = ledger
        .stages
        .iter()
        .position(|s| s.retained["code_admitted"] == true)
        .unwrap();
    let path = root.join(&ledger.stages[slot].file);
    let mut attempt: Attempt = read(&path)?;
    attempt.report.as_mut().unwrap().endpoints[0]
        .training
        .model
        .coefficients[0] += 0.1;
    overwrite(&path, &attempt);
    ledger.stages[slot] = ledger::stage(&root, slot, &attempt)?;
    overwrite(&root.join("ledger.json"), &ledger);
    assert!(
        execute::replay_with(&f.destination, name, &inv, &parent, input(name, false)?).is_err()
    );
    Ok(())
}

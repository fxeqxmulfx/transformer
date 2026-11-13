use super::*;
#[test]
fn selector_rejects_missing_scope_and_preserves_earlier_ce_ties() -> Result<()> {
    let (parent, names) = parent_value()?;
    let offer = comparison::Offer {
        task: names[0].clone(),
        mode: training::Mode::Full,
        train_ce: 1.,
        model_sha256: "a".repeat(64),
        report_sha256: "b".repeat(64),
    };
    let selected = comparison::select(&parent, std::slice::from_ref(&offer), &names)?;
    assert_eq!(
        selected["arms"][0]["choices"][0]["selected"],
        parent["arms"][0]["choices"][0]["reference"]
    );
    for i in 0..8 {
        let mut bad = parent.clone();
        match i {
            0 => {
                bad["arms"].as_array_mut().unwrap().pop();
            }
            1 => {
                bad["arms"][0]["choices"].as_array_mut().unwrap().pop();
            }
            2 => bad["arms"][0]["choices"][0]["task"] = json!("unknown"),
            3 => bad["arms"][0]["choices"][0]["train_ce"] = json!(-1),
            4 => {
                bad["complete_parent"]["depth_outcomes"]
                    .as_array_mut()
                    .unwrap()
                    .pop();
            }
            5 => {
                bad["complete_parent"]["transformer_endpoints"]
                    .as_array_mut()
                    .unwrap()
                    .pop();
            }
            6 => bad["choose_between_initializers"] = json!(true),
            _ => bad["constructor_accounting"]["charged_once_across_both_arms"] = json!(false),
        }
        assert!(comparison::select(&bad, &[], &names).is_err());
    }
    assert!(comparison::select(&parent, &[offer.clone(), offer], &names).is_err());
    Ok(())
}
#[test]
fn original_state_metrics_masks_and_selection_mutations_fail_replay() -> Result<()> {
    let f = Fixture::new()?;
    let (inv, parent) = f.prepare()?;
    f.successful(0, &inv, &parent)?;
    let dir = f.run.join("fixture-0");
    let original: Value = read(&dir.join("result.json"))?;
    for i in 0..6 {
        let mut bad = original.clone();
        match i {
            0 => bad["evaluation"][0]["metrics"]["test"]["sequence_accuracy"] = json!(0.0),
            1 => bad["evaluation"][0]["model_sha256"] = json!("a".repeat(64)),
            2 => bad["evaluation"][0]["unknown_prefixes"]["train"]["known_mask"][0] = json!(false),
            3 => bad["evaluation"][0]["audit"]["train_ce"] = json!(1.),
            4 => bad["selection"]["heldout_inputs"] = json!(true),
            _ => bad["unseen_identification"] = json!(true),
        }
        overwrite(&dir.join("result.json"), &bad);
        assert!(execute::replay_with(&f.run, "fixture-0", &inv, &parent, input(0)?).is_err());
    }
    overwrite(&dir.join("result.json"), &original);
    let mut x = input(0)?;
    x.configuration
        .as_mut()
        .unwrap()
        .discovery
        .input
        .observations[0]
        .label = 0;
    assert!(execute::replay_with(&f.run, "fixture-0", &inv, &parent, x).is_err());
    let original: execute::Attempt = read(&dir.join("attempt.json"))?;
    let mut changed = original.clone();
    changed.report.as_mut().unwrap().endpoints[0]
        .training
        .as_mut()
        .unwrap()
        .work
        .gradient_examples += 1;
    overwrite(&dir.join("attempt.json"), &changed);
    assert!(execute::replay_with(&f.run, "fixture-0", &inv, &parent, input(0)?).is_err());
    Ok(())
}
#[test]
fn parent_review_snapshots_and_complete_artifacts_are_bound() -> Result<()> {
    let f = Fixture::new()?;
    let (_inv, _) = f.prepare()?;
    let review = &f.request.parent.review.exit;
    fs::write(review, "1\n")?;
    assert!(inventory::inventory_with(&f.run, &f.suite, false).is_err());
    fs::write(review, "0\n")?;
    let file = f.run.join("parent/summary.json");
    let original = fs::read(&file)?;
    fs::write(&file, b"{}")?;
    assert!(inventory::inventory_with(&f.run, &f.suite, false).is_err());
    fs::write(file, original)?;
    overwrite(
        &f.request.parent.root.join("old-model.json"),
        &json!({"changed":true}),
    );
    assert!(inventory::inventory_with(&f.run, &f.suite, true).is_err());
    Ok(())
}

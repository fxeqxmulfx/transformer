use super::{fixtures::*, *};
fn pooled(name: &str) -> Result<inputs::Inputs> {
    let mut t = task(true);
    t.vocab = 101;
    for e in t.splits.values_mut().flatten() {
        let old = e.targets[1] - 3;
        e.targets[1] = 1 + old / 2 + 97 * (old % 2);
        let base = e.tokens[0];
        e.field = Some((base, (e.targets[1] - 50 + base).rem_euclid(97)));
    }
    inputs::assemble(t, name, json!({"synthetic":true}))
}
#[test]
fn colliding_raw_classes_use_unchanged_pooled_metric_and_full_vocabulary_ce() -> Result<()> {
    let mut f = Fixture::new()?;
    f.planned.truncate(1);
    let name = &f.planned[0];
    let x = pooled(name)?;
    inventory::prepare_with(
        &f.destination,
        f.request.clone(),
        &f.suite,
        &f.planned,
        vec![x.record.clone()],
    )?;
    let (inv, parent) = inventory::inventory_with(&f.destination, &f.suite, &f.planned, true)?;
    execute::run_with(
        &f.destination,
        name,
        &inv,
        &parent,
        x,
        std::time::Instant::now(),
    )?;
    let replay = execute::replay_with(&f.destination, name, &inv, &parent, pooled(name)?)?;
    assert_eq!(replay["all_actual_models_and_metrics_replayed"], true);
    let result: Value = read(&f.destination.join(name).join("result.json"))?;
    assert_eq!(result["eligible_offers"], 3);
    let candidates = result["evaluation"].as_array().unwrap();
    assert_eq!(candidates.len(), 6);
    for e in candidates.iter().filter(|e| e["offer_eligible"] == true) {
        assert_eq!(e["prime"], 3);
        assert_eq!(e["all_splits_pass"], true);
        assert_eq!(e["metrics"]["validation"]["sequence_accuracy"], 1.);
        assert_eq!(e["metrics"]["validation"]["native_token_accuracy"], 0.);
        assert!(e["train_ce"].as_f64().unwrap() > 0.6);
        assert_eq!(e["raw_label_identification"], false);
    }
    Ok(())
}
#[test]
fn summary_rejects_resealed_ineligible_models_scope_and_success_claims() -> Result<()> {
    let f = Fixture::new()?;
    let (inv, parent) = f.prepare(true)?;
    f.successful(&inv, &parent, true)?;
    f.failed()?;
    let name = &f.planned[0];
    let file = f.destination.join(name).join("result.json");
    let replay_path = f.destination.join(format!("{name}.replay.json"));
    let saved: Value = read(&file)?;
    let replay: Value = read(&replay_path)?;
    for mutation in 0..8 {
        let mut bad = saved.clone();
        match mutation {
            0 => bad["evaluation"][0]["model_sha256"] = json!("c".repeat(64)),
            1 => bad["evaluation"][0]["offer_eligible"] = json!(true),
            2 => bad["evaluation"][0]["primary_split"] = json!("train"),
            3 => bad["evaluation"][0]["raw_label_identification"] = json!(true),
            4 => bad["evaluation"][0]["all_splits_pass"] = json!(true),
            5 => bad["evaluation"][0] = bad["evaluation"][1].clone(),
            6 => bad["all55_stages_retained"] = json!(false),
            _ => bad["evaluation"][0]["metrics"]["train"]["loss"] = json!(0.),
        }
        overwrite(&file, &bad);
        let mut resealed = replay.clone();
        resealed["result_sha256"] = json!(digest(&file)?);
        overwrite(&replay_path, &resealed);
        assert!(
            summary::summarize_with(&f.destination, &inv, parent.clone(), &f.planned).is_err(),
            "mutation{mutation}"
        );
    }
    overwrite(&file, &saved);
    overwrite(&replay_path, &replay);
    assert!(summary::summarize_with(&f.destination, &inv, parent, &f.planned).is_ok());
    Ok(())
}
#[test]
fn failed_replay_retains_its_model_evidence_and_duplicate_inputs_are_rejected() -> Result<()> {
    let f = Fixture::new()?;
    let (inv, parent) = f.prepare(false)?;
    f.successful(&inv, &parent, false)?;
    f.failed()?;
    let name = &f.planned[0];
    fs::write(f.destination.join(format!("{name}.replay.exit")), "1\n")?;
    fs::write(
        f.destination.join(format!("{name}.replay.log")),
        "synthetic numerical replay failure\n",
    )?;
    fs::write(
        f.destination.join(format!("{name}.log")),
        "constructor completed\n",
    )?;
    let out = summary::summarize_with(&f.destination, &inv, parent.clone(), &f.planned)?;
    assert_eq!(out["process_failures"], 2);
    assert_eq!(
        out["records"][0]["failed_replay_evidence"]
            .as_array()
            .unwrap()
            .len(),
        3
    );
    assert!(
        out["records"][0]["partial_artifacts"]
            .as_array()
            .unwrap()
            .len()
            > 55
    );
    for arm in 0..2 {
        assert_eq!(out["arms"][arm]["choices"], parent["arms"][arm]["choices"]);
    }
    let mut bad = inv.clone();
    bad["inputs"][1] = bad["inputs"][0].clone();
    overwrite(&f.destination.join("inventory.json"), &bad);
    assert!(inventory::inventory_with(&f.destination, &f.suite, &f.planned, false).is_err());
    let mut bad = inv["parent_snapshots"].clone();
    bad[0]["file"] = json!("/");
    assert!(parent::check_snapshot(&f.destination.join("parent"), &bad).is_err());
    Ok(())
}

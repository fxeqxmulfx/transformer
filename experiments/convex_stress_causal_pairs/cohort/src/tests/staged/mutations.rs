use super::*;
fn reseal(root: &Path, inv: &Value) -> Result<staging::Reference> {
    let result = staging::record(&root.join("fixture-0"), inv, &input(0)?.record)?;
    overwrite(&root.join("fixture-0/result.json"), &result);
    // Deliberately forged outer seals: the independent numeric import must
    // still reject changed states even when every local hash is recomputed.
    overwrite(
        &root.join("fixture-0.replay.json"),
        &staging::seal(root, "fixture-0")?,
    );
    overwrite(
        &root.join("summary.json"),
        &staging::summarize_with(root, inv)?,
    );
    Ok(staging::Reference {
        root: root.into(),
        summary_sha256: digest(&root.join("summary.json"))?,
        inventory_sha256: digest(&root.join("inventory.json"))?,
    })
}
#[test]
fn rehashed_actual_state_and_work_mutations_fail_before_selection() -> Result<()> {
    for variant in 0..5 {
        let mut f = Fixture::new()?;
        let (root, inv, _) = complete(&f)?;
        let path = root.join("fixture-0/attempt.json");
        let mut a: execute::Attempt = read(&path)?;
        let r = a.report.as_mut().unwrap();
        match variant {
            0 => r.endpoints[0].training.as_mut().unwrap().model.coefficients[0] += 0.25,
            1 => {
                r.endpoints[0]
                    .training
                    .as_mut()
                    .unwrap()
                    .work
                    .gradient_examples += 1
            }
            2 => r.discovery.groups[0].decoder.as_mut().unwrap().labels[0] = 19,
            3 => r.endpoints[0].training.as_mut().unwrap().initial_ce += 1.,
            _ => r.initialization.as_mut().unwrap().ce_gradients += 1,
        }
        overwrite(&path, &a);
        f.request.staging = Some(reseal(&root, &inv)?);
        let (joined, parent) = f.prepare()?; // All outer hashes now pass.
        assert!(execute::run_with(&f.run, "fixture-0", &joined, &parent, input(0)?).is_err());
        assert!(!f.run.join("fixture-0/selection.json").exists());
        assert!(!f.run.join("fixture-0/result.json").exists());
    }
    Ok(())
}
#[test]
fn stage_pins_reject_changed_timing_resources_and_extra_partial_artifacts() -> Result<()> {
    let mut f = Fixture::new()?;
    let (root, stage, pin) = complete(&f)?;
    f.request.staging = Some(pin);
    let (inv, parent) = f.prepare()?;
    f.successful(0, &inv, &parent)?;
    let names = [
        "fixture-0/attempt.json",
        "fixture-0/result.json",
        "fixture-0.replay.json",
        "summary.json",
        "fixture-0.run.resources",
    ];
    let saved = names
        .iter()
        .map(|n| fs::read(root.join(n)))
        .collect::<std::io::Result<Vec<_>>>()?;
    for variant in 0..3 {
        match variant {
            0 => {
                let mut a: execute::Attempt = read(&root.join(names[0]))?;
                a.construction_seconds += 1.;
                overwrite(&root.join(names[0]), &a);
                reseal(&root, &stage)?; // The externally pinned summary cannot be changed.
            }
            1 => fs::write(root.join(names[4]), "changed resource evidence")?,
            _ => fs::write(
                root.join("fixture-1/unrecorded-partial"),
                "changed failed-stage evidence",
            )?,
        }
        assert!(inventory::inventory_with(&f.run, &f.suite, true).is_err());
        if variant == 0 || variant == 1 {
            assert!(execute::replay_with(&f.run, "fixture-0", &inv, &parent, input(0)?).is_err());
        }
        for (name, bytes) in names.iter().zip(&saved) {
            fs::write(root.join(name), bytes)?;
        }
        if variant == 2 {
            fs::remove_file(root.join("fixture-1/unrecorded-partial"))?;
        }
        assert!(inventory::inventory_with(&f.run, &f.suite, true).is_ok());
    }
    Ok(())
}
#[test]
fn staged_inputs_scope_source_and_original_parent_review_are_required() -> Result<()> {
    let mut f = Fixture::new()?;
    let (root, inv, pin) = complete(&f)?;
    for variant in 0..5 {
        let mut bad = inv.clone();
        match variant {
            0 => {
                bad["tasks"].as_array_mut().unwrap().pop();
            }
            1 => bad["source_fingerprint"] = json!("0".repeat(64)),
            2 => bad["binary_sha256"] = json!("0".repeat(64)),
            3 => bad["model_protocol"]["training"][0]["updates"] = json!(0),
            _ => bad["heldout_evaluation"] = json!(true),
        }
        overwrite(&root.join("inventory.json"), &bad);
        assert!(staging::inventory_with(&root, &f.suite).is_err());
    }
    overwrite(&root.join("inventory.json"), &inv);
    let mut changed = input(0)?;
    changed
        .configuration
        .as_mut()
        .unwrap()
        .discovery
        .input
        .observations[0]
        .label = 19;
    assert!(staging::replay_with(&root, "fixture-0", &inv, changed).is_err());
    f.request.staging = Some(pin);
    fs::write(&f.request.parent.review.exit, "1\n")?;
    assert!(f.prepare().is_err());
    assert!(!f.run.exists());
    fs::write(&f.request.parent.review.exit, "0\n")?;
    f.prepare()?;
    Ok(())
}
#[test]
fn a_failed_training_replay_is_retained_and_cannot_be_imported() -> Result<()> {
    let mut f = Fixture::new()?;
    let (root, inv, _) = complete(&f)?;
    fs::write(root.join("fixture-0.replay.exit"), "1\n")?;
    let summary = staging::summarize_with(&root, &inv)?;
    assert_eq!(summary["records"][0]["status"], "process_failure");
    assert!(summary["records"][0]["artifacts"].as_array().unwrap().len() > 7);
    assert_eq!(summary["constructor_accounting"]["process_failures"], 34);
    overwrite(&root.join("summary.json"), &summary);
    f.request.staging = Some(staging::Reference {
        root: root.clone(),
        summary_sha256: digest(&root.join("summary.json"))?,
        inventory_sha256: digest(&root.join("inventory.json"))?,
    });
    let (joined, parent) = f.prepare()?;
    assert!(execute::run_with(&f.run, "fixture-0", &joined, &parent, input(0)?).is_err());
    let imported: Value = read(&f.run.join("fixture-0/import.json"))?;
    assert_eq!(imported["record"]["status"], "process_failure");
    assert!(!f.run.join("fixture-0/selection.json").exists());
    Ok(())
}

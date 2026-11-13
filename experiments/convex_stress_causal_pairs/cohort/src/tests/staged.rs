use super::*;
mod mutations;
pub(super) fn prepared(f: &Fixture) -> Result<(PathBuf, Value)> {
    let root = f.temp.join("training");
    let records = (0..35)
        .map(|i| Ok(input(i)?.record))
        .collect::<Result<_>>()?;
    staging::prepare_with(
        &root,
        staging::Request {
            producer_commit: f.request.producer_commit.clone(),
            suite_sha256: hash(SUITE.as_bytes()),
        },
        &f.suite,
        records,
    )?;
    fs::write(
        root.join("prepare.resources"),
        "synthetic preparation resource measurement\n",
    )?;
    Ok((root.clone(), staging::inventory_with(&root, &f.suite)?))
}
pub(super) fn successful(root: &Path, inv: &Value, i: usize) -> Result<()> {
    let task = format!("fixture-{i}");
    staging::run_with(root, &task, inv, input(i)?)?;
    let replay = staging::replay_with(root, &task, inv, input(i)?)?;
    write(&root.join(format!("{task}.replay.json")), &replay)?;
    for stage in ["run", "replay"] {
        fs::write(root.join(format!("{task}.{stage}.exit")), "0\n")?;
        fs::write(
            root.join(format!("{task}.{stage}.resources")),
            "synthetic measured resources\n",
        )?;
    }
    fs::write(root.join(format!("{task}.log")), "")?;
    fs::write(root.join(format!("{task}.replay.log")), "")?;
    Ok(())
}
pub(super) fn failed(root: &Path, i: usize) -> Result<()> {
    let task = format!("fixture-{i}");
    fs::create_dir_all(root.join(&task))?;
    write(
        &root.join(&task).join("partial.json"),
        &json!({"retained_work":123,"failed":true}),
    )?;
    fs::write(root.join(format!("{task}.run.exit")), "137\n")?;
    fs::write(
        root.join(format!("{task}.run.resources")),
        "synthetic failed process resources\n",
    )?;
    fs::write(
        root.join(format!("{task}.log")),
        "synthetic process failure\n",
    )?;
    Ok(())
}
pub(super) fn finish(root: &Path, inv: &Value) -> Result<staging::Reference> {
    write(
        &root.join("summary.json"),
        &staging::summarize_with(root, inv)?,
    )?;
    Ok(staging::Reference {
        root: root.into(),
        summary_sha256: digest(&root.join("summary.json"))?,
        inventory_sha256: digest(&root.join("inventory.json"))?,
    })
}
pub(super) fn complete(f: &Fixture) -> Result<(PathBuf, Value, staging::Reference)> {
    let (root, inv) = prepared(f)?;
    successful(&root, &inv, 0)?;
    successful(&root, &inv, 2)?;
    for i in (1..35).filter(|&i| i != 2) {
        failed(&root, i)?;
    }
    let pin = finish(&root, &inv)?;
    Ok((root, inv, pin))
}
#[test]
fn staging_replays_without_a_parent_and_never_scores_heldout_answers() -> Result<()> {
    let f = Fixture::new()?;
    let (root, inv) = prepared(&f)?;
    // Remove even the synthetic comparison parent: staging has no dependency on it.
    fs::remove_dir_all(&f.request.parent.root)?;
    successful(&root, &inv, 0)?;
    let result: Value = read(&root.join("fixture-0/result.json"))?;
    assert_eq!(result["parent_comparison"], false);
    assert_eq!(result["heldout_evaluation"], false);
    assert!(result.get("selection").is_none() && result.get("evaluation").is_none());
    let mut changed = input(0)?;
    for e in changed.task.splits.get_mut("test").unwrap() {
        e.targets.fill(0);
        e.tokens.resize(99, 1);
    }
    let original = input(0)?;
    let a = execute::construct(&original.configuration, &original.refusal);
    let b = execute::construct(&changed.configuration, &changed.refusal);
    assert_eq!(a.report, b.report);
    assert_eq!(
        staging::replay_with(&root, "fixture-0", &inv, changed)?,
        staging::seal(&root, "fixture-0")?
    );
    assert!(staging::summarize_with(&root, &inv).is_err()); // Missing33 failures cannot disappear.
    Ok(())
}
#[test]
fn staged_import_matches_fresh_models_choices_metrics_and_masks() -> Result<()> {
    let fresh = Fixture::new()?;
    let (fresh_inv, parent) = fresh.prepare()?;
    fresh.successful(0, &fresh_inv, &parent)?;
    let mut f = Fixture::new()?;
    let (_, _, pin) = complete(&f)?;
    f.request.staging = Some(pin);
    let (inv, p) = f.prepare()?;
    f.successful(0, &inv, &p)?;
    let a: execute::Attempt = read(&fresh.run.join("fixture-0/attempt.json"))?;
    let b: execute::Attempt = read(&f.run.join("fixture-0/attempt.json"))?;
    assert_eq!(a.report, b.report);
    let mut a: Value = read(&fresh.run.join("fixture-0/result.json"))?;
    let mut b: Value = read(&f.run.join("fixture-0/result.json"))?;
    assert_eq!(a["construction_reused"], false);
    assert_eq!(b["construction_reused"], true);
    // Provenance hashes include distinct measured construction times. The
    // actual model, selection scores and every held-out metric/mask are equal.
    for value in [&mut a, &mut b] {
        for row in value["evaluation"].as_array_mut().unwrap() {
            let object = row.as_object_mut().unwrap();
            object.remove("reference");
            object.remove("report_sha256");
        }
    }
    assert_eq!(a["evaluation"], b["evaluation"]);
    for (a, b) in a["selection"]["selection"]["arms"]
        .as_array()
        .unwrap()
        .iter()
        .zip(b["selection"]["selection"]["arms"].as_array().unwrap())
    {
        for (x, y) in a["choices"]
            .as_array()
            .unwrap()
            .iter()
            .zip(b["choices"].as_array().unwrap())
        {
            let normalize = |row: &Value| {
                let mut row = row.clone();
                if row["selected"] != row["prior_reference"] {
                    assert!(
                        row["candidates"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .any(|c| c["source"] == "causal_pairs"
                                && c["reference"] == row["selected"])
                    );
                    row["selected"] = json!("new_actual_causal_endpoint");
                }
                for c in row["candidates"].as_array_mut().unwrap() {
                    if c["source"] == "causal_pairs" {
                        c.as_object_mut().unwrap().remove("reference");
                        c.as_object_mut().unwrap().remove("report_sha256");
                    }
                }
                row
            };
            assert_eq!(normalize(x), normalize(y));
        }
    }
    Ok(())
}
#[test]
fn failed_imports_keep_all35_parents_and_charge_all_staged_construction() -> Result<()> {
    let mut f = Fixture::new()?;
    let (_, _, pin) = complete(&f)?;
    f.request.staging = Some(pin);
    let (inv, parent) = f.prepare()?;
    // A later metric refusal cannot erase a successfully staged model's cost.
    let mut bad_metrics = input(0)?;
    bad_metrics.task.splits.get_mut("test").unwrap()[0].limit = 5000;
    assert!(execute::run_with(&f.run, "fixture-0", &inv, &parent, bad_metrics).is_err());
    failed(&f.run, 0)?;
    f.successful(2, &inv, &parent)?;
    for i in (1..35).filter(|&i| i != 2) {
        assert!(
            execute::run_with(&f.run, &format!("fixture-{i}"), &inv, &parent, input(i)?).is_err()
        );
        failed(&f.run, i)?;
    }
    let summary = summary::summarize_with(&f.run, &inv, parent.clone())?;
    assert_eq!(summary["records"].as_array().unwrap().len(), 35);
    assert_eq!(summary["process_failures"], 34);
    assert_eq!(
        summary["constructor_accounting"]["complete_gradient_batches"],
        272
    );
    assert_eq!(
        summary["constructor_accounting"]["complete_constructor_seconds"],
        inv["staging"]["summary"]["constructor_accounting"]["complete_constructor_seconds"]
    );
    assert_eq!(summary["staged_construction"], inv["staging"]);
    assert_eq!(summary["complete_parent"], parent);
    for arm in summary["arms"].as_array().unwrap() {
        assert_eq!(arm["choices"].as_array().unwrap().len(), 105);
        let old = parent["arms"]
            .as_array()
            .unwrap()
            .iter()
            .find(|a| a["initializer"] == arm["initializer"])
            .unwrap();
        assert_eq!(arm["choices"], old["choices"]);
    }
    Ok(())
}

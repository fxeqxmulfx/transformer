use super::*;
use std::{
    collections::BTreeMap,
    time::{Instant, SystemTime, UNIX_EPOCH},
};
fn fixture(c: &Case, constant: bool) -> Result<Inputs> {
    let (truth, mut observations) = crate::tests::fixture();
    if constant {
        observations = vec![Sample { cell: 0, label: 1 }, Sample { cell: 1, label: 1 }];
    }
    let example = |cell: usize, label: usize| data::Example {
        task: c.task.clone(),
        tokens: vec![(cell / truth.n) as i64, (cell % truth.n) as i64],
        targets: vec![-100, label as i64],
        changes: vec![false, true],
        prompt: vec![],
        answer: vec![],
        limit: 0,
        field: None,
        row_index: None,
    };
    let task = data::Task {
        vocab: truth.vocab,
        selection: "validation".into(),
        splits: BTreeMap::from([
            (
                "train".into(),
                observations
                    .iter()
                    .map(|s| example(s.cell, s.label))
                    .collect(),
            ),
            (
                "validation".into(),
                (0..truth.n * truth.n)
                    .filter(|cell| cell % 17 == 0)
                    .map(|cell| example(cell, truth.output_labels[cell]))
                    .collect(),
            ),
            (
                "test".into(),
                (0..truth.n * truth.n)
                    .map(|cell| example(cell, truth.output_labels[cell]))
                    .collect(),
            ),
        ]),
    };
    Ok(Inputs {
        task,
        row: json!({"data_fingerprints":{"train":"fixture","validation":"withheld-fixture"}}),
        observations,
        explicit_bytes: memory_bound(truth.n, truth.q, truth.vocab)?,
    })
}
fn temporary() -> Result<std::path::PathBuf> {
    let root = std::env::temp_dir().join(format!(
        "decoded-cohort-{}-{}",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    ));
    fs::create_dir(&root)?;
    Ok(root)
}
fn fixture_inventory(c: &Case, x: &Inputs) -> Result<Value> {
    Ok(
        json!({"source_fingerprint":fingerprint(),"protocol":protocol()?,"cases":cases()?,
        "producer_commit":"0000000000000000000000000000000000000000","parents":{},
        "tasks":[task_record(c,x)]}),
    )
}
#[test]
fn original_inventory_and_frozen_parent_are_exact_and_missing_parent_stays_stopped() -> Result<()> {
    assert_eq!(cases()?.len(), 8);
    assert_eq!(
        cases()?
            .iter()
            .map(Case::id)
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        8
    );
    assert_eq!(
        convex_phase_start::cohort::fingerprint(),
        protocol()?["parent_fingerprint"]
    );
    assert_eq!(protocol()?["options"]["code_trials"], 16);
    assert_eq!(protocol()?["options"]["power_steps"], 64);
    assert_eq!(protocol()?["options"]["refinement_steps"], 24);
    let root = temporary()?;
    assert!(prepare(&root.join("new"), &root.join("absent-parent")).is_err());
    assert!(!root.join("new").exists());
    let large = root.join("large.json");
    fs::File::create(&large)?.set_len(1025)?;
    assert!(bytes(&large, 1024).is_err());
    fs::remove_dir_all(root)?;
    Ok(())
}
#[test]
fn saved_decoded_fixture_replays_and_rejects_forged_coordinates_costs_and_metrics() -> Result<()> {
    let root = temporary()?;
    let c = cases()?.remove(0);
    let x = fixture(&c, false)?;
    let inv = fixture_inventory(&c, &x)?;
    write_new(&root.join("inventory.json"), &inv)?;
    run::execute(&root, c.clone(), inv.clone(), x, Instant::now())?;
    assert!(
        run::execute(
            &root,
            c.clone(),
            inv.clone(),
            fixture(&c, false)?,
            Instant::now()
        )
        .is_err()
    );
    let out = root.join(c.id());
    let result_path = out.join("result.json");
    let model_path = out.join("model.json");
    let proposal_path = out.join("proposal.json");
    let original_result = fs::read(&result_path)?;
    let original_model = fs::read(&model_path)?;
    let original_proposal = fs::read(&proposal_path)?;
    let r: Value = serde_json::from_slice(&original_result)?;
    assert_eq!(r["status"], "complete");
    assert_eq!(r["metrics"]["validation"]["sequence_accuracy"], 1.);
    assert_eq!(r["proposal_exact_fit"], true);
    let seal = replay::replay_with(&root, c.clone(), inv.clone(), fixture(&c, false)?)?;
    assert_eq!(seal["audit"]["raw_latent_coordinate_audit"], true);
    assert_eq!(seal["audit"]["constructor_reproduced"], true);
    assert_eq!(seal["result_sha256"], hash(&original_result));
    write_new(&root.join(format!("{}.replay.json", c.id())), &seal)?;
    summary::validate_record(&root, &c, &inv)?;

    let mut changed: Model = serde_json::from_slice(&original_model)?;
    changed.output_labels[0] = (changed.output_labels[0] + 1) % changed.vocab;
    fs::write(&model_path, serde_json::to_vec(&changed)?)?;
    let mut forged = r.clone();
    forged["model_sha256"] = hash(&fs::read(&model_path)?).into();
    fs::write(&result_path, serde_json::to_vec(&forged)?)?;
    assert!(replay::replay_with(&root, c.clone(), inv.clone(), fixture(&c, false)?).is_err());
    fs::write(&model_path, original_model)?;

    let mut changed: Value = serde_json::from_slice(&original_proposal)?;
    changed["cost"]["sketch_observations"] = 0.into();
    fs::write(&proposal_path, serde_json::to_vec(&changed)?)?;
    let mut forged = r.clone();
    forged["proposal_sha256"] = hash(&fs::read(&proposal_path)?).into();
    forged["proposal_cost"] = changed["cost"].clone();
    fs::write(&result_path, serde_json::to_vec(&forged)?)?;
    assert!(replay::replay_with(&root, c.clone(), inv.clone(), fixture(&c, false)?).is_err());
    fs::write(&proposal_path, original_proposal)?;

    let mut forged = r.clone();
    forged["metrics"]["validation"]["sequence_accuracy"] = 0.into();
    fs::write(&result_path, serde_json::to_vec(&forged)?)?;
    assert!(summary::validate_record(&root, &c, &inv).is_err());
    assert!(replay::replay_with(&root, c.clone(), inv.clone(), fixture(&c, false)?).is_err());
    fs::write(&result_path, original_result)?;
    replay::replay_with(&root, c.clone(), inv, fixture(&c, false)?)?;
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn imperfect_covered_endpoint_is_saved_and_replayed_with_its_errors() -> Result<()> {
    let root = temporary()?;
    let c = cases()?.remove(0);
    let (truth, _) = crate::tests::fixture();
    let mut x = fixture(&c, false)?;
    let mut training = x.task.splits["test"].clone();
    // Every permuted-phase row has the same class histogram on a complete grid.
    // Change one row's histogram so this search family cannot interpolate it.
    training[0].targets[1] = ((truth.output_labels[0] + 1) % truth.vocab) as i64;
    x.observations = training
        .iter()
        .enumerate()
        .map(|(cell, e)| Sample {
            cell,
            label: e.targets[1] as usize,
        })
        .collect();
    x.task.splits.insert("train".into(), training);
    let inv = fixture_inventory(&c, &x)?;
    let replay_x = Inputs {
        task: data::Task {
            vocab: x.task.vocab,
            selection: x.task.selection.clone(),
            splits: x.task.splits.clone(),
        },
        row: x.row.clone(),
        observations: x.observations.clone(),
        explicit_bytes: x.explicit_bytes,
    };
    run::execute(&root, c.clone(), inv.clone(), x, Instant::now())?;
    let r = read(&root.join(c.id()).join("result.json"))?;
    assert_eq!(r["status"], "complete");
    assert_eq!(r["proposal_exact_fit"], false);
    assert!(number(&r, "proposal_best_errors")? > 0.);
    assert_eq!(r["proposal_cost"]["code_trials"], 16);
    let seal = replay::replay_with(&root, c, inv, replay_x)?;
    assert_eq!(seal["audit"]["exact_observed_fit"], false);
    assert_eq!(seal["audit"]["raw_latent_coordinate_audit"], true);
    fs::remove_dir_all(root)?;
    Ok(())
}
#[test]
fn failed_proposal_trace_replays_without_substituting_a_model() -> Result<()> {
    let root = temporary()?;
    let c = cases()?.remove(0);
    let x = fixture(&c, true)?;
    let inv = fixture_inventory(&c, &x)?;
    run::execute(&root, c.clone(), inv.clone(), x, Instant::now())?;
    let r = read(&root.join(c.id()).join("result.json"))?;
    assert_eq!(r["status"], "proposal_ineligible");
    assert_eq!(r["proposal_cost"]["code_trials"], 16);
    assert!(!root.join(c.id()).join("model.json").exists());
    let seal = replay::replay_with(&root, c.clone(), inv, fixture(&c, true)?)?;
    assert_eq!(seal["audit"]["rejected_proposal_reproduced"], true);
    fs::remove_dir_all(root)?;
    Ok(())
}

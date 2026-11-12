use super::{integrate::combine, *};
use crate::{projection::Projector, solver::Solver, tests::folded_training};
use std::{
    collections::BTreeMap,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

const LABELS: [usize; 9] = [0, 1, 2, 1, 0, 1, 2, 1, 0];
fn example(cell: usize, label: usize) -> data::Example {
    data::Example {
        task: "paired-fixture".into(),
        tokens: vec![(cell / 3) as i64, (cell % 3) as i64],
        targets: vec![-100, label as i64],
        changes: vec![false, true],
        prompt: vec![],
        answer: vec![],
        limit: 0,
        field: None,
        row_index: None,
    }
}
/// Lean `foldedLabels` with the cell (2, 1) withheld from training.
fn fixture() -> Result<Inputs> {
    let train = folded_training()
        .iter()
        .map(|s| example(s.cell, s.label))
        .collect();
    let task = data::Task {
        vocab: 3,
        selection: "validation".into(),
        splits: BTreeMap::from([
            ("train".into(), train),
            ("validation".into(), vec![example(5, LABELS[5])]),
            (
                "test".into(),
                (0..9).map(|cell| example(cell, LABELS[cell])).collect(),
            ),
        ]),
    };
    assemble(
        task,
        json!({"case":{"batch":3,"observe":2},"data_fingerprints":{"fixture":"paired-withheld-v1"}}),
    )
}
fn temporary() -> Result<std::path::PathBuf> {
    let root = std::env::temp_dir().join(format!(
        "paired-cohort-{}-{}",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    ));
    fs::create_dir(&root)?;
    Ok(root)
}
fn planned(arm: &str) -> Result<Case> {
    Ok(cases()?.into_iter().find(|c| c.arm == arm).unwrap())
}
fn fixture_inventory(c: &Case) -> Result<Value> {
    Ok(
        json!({"source_fingerprint":fingerprint(),"protocol":protocol()?,"cases":cases()?,
        "producer_commit":"0000000000000000000000000000000000000000",
        "tasks":[task_record(&c.task, &fixture()?)]}),
    )
}

#[test]
fn inventory_is_frozen_and_original_tasks_have_the_planned_sizes() -> Result<()> {
    let all = cases()?;
    assert_eq!(all.len(), 24);
    let ids: std::collections::BTreeSet<_> = all.iter().map(Case::id).collect();
    assert_eq!(ids.len(), 24);
    assert!(ids.contains("minibatch--arithmetic-abs-diff--fold--rank_one"));
    assert!(case("full_batch--arithmetic-abs-diff--fold--full_rank").is_err());
    let p = protocol()?;
    assert_eq!(
        convex_stress_decode::cohort::fingerprint(),
        p["parent_fingerprint"]
    );
    assert_eq!(
        convex_stress_decode::cohort::protocol()?["tasks"],
        p["tasks"]
    );
    assert_eq!(
        (p["full_updates"].clone(), p["minibatch_updates"].clone()),
        (json!(128), json!(512))
    );
    assert_eq!(p["forward_updates"], 128);
    for c in all.iter().filter(|c| c.arm == "minibatch") {
        let x = inputs(c)?;
        assert_eq!(
            x.shape,
            Shape {
                side: 97,
                orbits: 97,
                vocab: 97
            }
        );
        assert_eq!(x.shape.latent_states(), 193);
        assert_eq!(x.decoder, Decoder::fold(97));
        assert!(x.explicit_bytes < 2 * 1024 * 1024 * 1024);
        assert!(x.observations.len() < 97 * 97);
        assert!(x.task.splits.len() >= 2 && x.task.splits.contains_key(&x.task.selection));
        let s = schedule(c, &x)?;
        assert_eq!((s.batch, s.cadence, s.limit), (256, 19, 512));
        assert_eq!(budgets(&c.budget, x.shape.side)?, (97., 97.));
    }
    let root = temporary()?;
    assert!(prepare(&root).is_err());
    let large = root.join("large.json");
    fs::File::create(&large)?.set_len(1025)?;
    assert!(bytes(&large, 1024).is_err());
    assert!(
        inputs(&Case {
            task: "arithmetic-half-sum".into(),
            ..planned("minibatch")?
        })
        .is_err()
    );
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn three_modes_run_replay_audit_and_summarize_on_the_fold_fixture() -> Result<()> {
    let root = temporary()?;
    for arm in ["full_batch", "minibatch", "forward_only"] {
        let c = planned(arm)?;
        let inv = fixture_inventory(&c)?;
        run::execute(&root, c.clone(), inv.clone(), fixture()?, Instant::now())?;
        assert!(run::execute(&root, c.clone(), inv.clone(), fixture()?, Instant::now()).is_err());
        let r = read(&root.join(c.id()).join("result.json"))?;
        assert_eq!(r["status"], "complete");
        assert_eq!(r["stop_reason"], "ideal_ce_excess", "{arm}");
        assert_eq!(r["shape"], json!({"side":3,"orbits":3,"vocab":3}));
        assert_eq!(r["latent_states"], 5);
        assert_eq!(r["metrics"]["train"]["sequence_accuracy"], 1.);
        // The withheld cell is recovered on this fixture; no theorem says it must be.
        assert_eq!(r["metrics"]["validation"]["sequence_accuracy"], 1.);
        assert_eq!(r["metrics"]["test"]["exact"], 9);
        assert_eq!(
            r["training_gradient_evaluations"] == 0,
            arm == "forward_only"
        );
        let trace = fs::read_to_string(root.join(c.id()).join("trace.jsonl"))?;
        assert_eq!(
            trace.lines().count() as u64,
            r["updates"].as_u64().unwrap() / 2 + 1
        );
        let seal = replay::replay_with(&root, c.clone(), inv.clone(), fixture()?)?;
        assert_eq!(seal["audit"]["raw_latent_coordinate_audit"], true);
        assert_eq!(seal["audit"]["frequencies"], 4);
        assert_eq!(seal["audit"]["diagnostics"]["test"]["correct"], 9);
        assert!(summary::validate_record(&root, &c, &inv).is_err());
        write_new(&root.join(format!("{}.replay.json", c.id())), &seal)?;
        summary::validate_record(&root, &c, &inv)?;
    }
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn replay_rejects_forged_tables_states_metrics_and_costs() -> Result<()> {
    let root = temporary()?;
    let c = planned("full_batch")?;
    let inv = fixture_inventory(&c)?;
    run::execute(&root, c.clone(), inv.clone(), fixture()?, Instant::now())?;
    let out = root.join(c.id());
    let paths = ["result.json", "model.json", "diagnostic_state.json"].map(|name| out.join(name));
    let original = [
        fs::read(&paths[0])?,
        fs::read(&paths[1])?,
        fs::read(&paths[2])?,
    ];
    let r: Value = serde_json::from_slice(&original[0])?;
    let attempt = || replay::replay_with(&root, c.clone(), inv.clone(), fixture()?);
    let seal = attempt()?;
    write_new(&root.join(format!("{}.replay.json", c.id())), &seal)?;
    let restore = || -> Result<()> {
        for (path, content) in paths.iter().zip(&original) {
            fs::write(path, content)?;
        }
        Ok(())
    };
    let forge = |r: &Value, key: &str, path: &Path| -> Result<()> {
        let mut forged = r.clone();
        forged[key] = hash(&fs::read(path)?).into();
        fs::write(&paths[0], serde_json::to_vec(&forged)?)?;
        Ok(())
    };
    let saved: Model = serde_json::from_slice(&original[1])?;

    // A decoded entry that the latent table does not produce.
    let mut changed = saved.clone();
    changed.output[0] -= 0.25;
    changed.output[1] += 0.25;
    fs::write(&paths[1], serde_json::to_vec(&changed)?)?;
    forge(&r, "model_sha256", &paths[1])?;
    assert!(attempt().is_err());
    restore()?;

    // A valid, feasible table that is not the endpoint of the saved splitting point.
    let uniform = Model::uniform(
        saved.shape,
        saved.decoder.clone(),
        saved.radius,
        saved.nuclear_budget,
    )?;
    fs::write(&paths[1], serde_json::to_vec(&uniform)?)?;
    forge(&r, "model_sha256", &paths[1])?;
    let error = attempt().unwrap_err().to_string();
    assert!(error.contains("saved splitting point"), "{error}");
    restore()?;

    // A decoder that is even but is not the vocabulary-only fold.
    let mut changed = saved.clone();
    changed.decoder.orbit_class = vec![0, 2, 1];
    changed.output = changed.decoder.decode(changed.shape, &changed.latent_half);
    fs::write(&paths[1], serde_json::to_vec(&changed)?)?;
    forge(&r, "model_sha256", &paths[1])?;
    let error = attempt().unwrap_err().to_string();
    assert!(error.contains("configuration"), "{error}");
    restore()?;

    let mut state: Vec<f64> = serde_json::from_slice(&original[2])?;
    state[4] += 0.125;
    fs::write(&paths[2], serde_json::to_vec(&state)?)?;
    forge(&r, "diagnostic_state_sha256", &paths[2])?;
    assert!(attempt().is_err());
    restore()?;

    for (pointer, value) in [
        ("/metrics/validation/sequence_accuracy", json!(0.)),
        ("/metrics/train/loss", json!(0.5)),
        ("/updates", json!(r["updates"].as_u64().unwrap() + 1)),
        ("/examples_presented", json!(1)),
        ("/costs/gradient_evaluations", json!(1)),
        ("/costs/decompositions", json!(1_000_000)),
        ("/stop_reason", json!("budget")),
        ("/shape/orbits", json!(4)),
        ("/moment_norms/0/nuclear", json!(0.)),
        ("/producer_commit", json!("1".repeat(40))),
    ] {
        let mut forged = r.clone();
        *forged.pointer_mut(pointer).unwrap() = value;
        fs::write(&paths[0], serde_json::to_vec(&forged)?)?;
        assert!(attempt().is_err(), "{pointer}");
        assert!(
            summary::validate_record(&root, &c, &inv).is_err(),
            "{pointer}"
        );
    }
    restore()?;
    assert_eq!(attempt()?, seal);
    summary::validate_record(&root, &c, &inv)?;
    fs::remove_dir_all(root)?;
    Ok(())
}

/// A report that the producer's own code would write for an arbitrary table.
fn report(model: &Model, x: &Inputs) -> Result<Value> {
    let solver = Solver::new(model, x.observations.clone(), Mode::Full)?;
    let metrics = metrics_all(model, &x.task)?;
    let excess = number(&metrics["train"], "loss")? - ideal_loss(model.shape.vocab);
    Ok(json!({"case":{"budget":"rank_one"},"metrics":metrics,
        "fit_defect":solver.fit_defect(model),"ideal_ce_excess":excess,
        "moment_norms":Projector::new(model.shape)?.norms(&model.latent_half)?}))
}

#[test]
fn raw_audit_alone_rejects_spectra_decoding_mass_and_metrics() -> Result<()> {
    let x = fixture()?;
    let (shape, decoder, half) = crate::tests::folded();
    let witness = Model::new(shape, decoder.clone(), 3., 3., half)?;
    let good = report(&witness, &x)?;
    let audit = audit::validate(&witness, &x.task, &good)?;
    assert_eq!(audit["diagnostics"]["train"]["one_hot_defect"], 0.);
    assert_eq!(audit["frequencies"], 4);

    // Zero on the diagonal and the orbit {2, -2} elsewhere: every row is an even
    // probability vector, but the first moment has nuclear norm about 4.24 > 3.
    let mut half = vec![0.; 27];
    for cell in 0..9 {
        if cell % 4 == 0 {
            half[cell * 3] = 1.;
        } else {
            half[cell * 3 + 2] = 0.5;
        }
    }
    let wide = Model::new(shape, decoder, 3., 3., half)?;
    assert!(wide.validate(&Projector::new(shape)?).is_err());
    let error = audit::validate(&wide, &x.task, &report(&wide, &x)?)
        .unwrap_err()
        .to_string();
    assert!(error.contains("raw spectral bound violation"), "{error}");

    let mut changed = witness.clone();
    changed.output[4] = 0.5;
    assert!(report(&changed, &x).is_err());
    let error = audit::validate(&changed, &x.task, &good)
        .unwrap_err()
        .to_string();
    assert!(error.contains("not the decoded latent table"), "{error}");

    let mut changed = witness.clone();
    changed.latent_half[1] = 0.25;
    let error = audit::validate(&changed, &x.task, &good)
        .unwrap_err()
        .to_string();
    assert!(error.contains("does not sum to one"), "{error}");

    for (pointer, value) in [
        ("/metrics/test/exact", json!(8)),
        ("/metrics/validation/loss", json!(1.)),
        ("/fit_defect", json!(0.25)),
        ("/ideal_ce_excess", json!(1e-3)),
        ("/moment_norms/1/operator", json!(2.5)),
        ("/case/budget", json!("full_rank")),
    ] {
        let mut forged = good.clone();
        *forged.pointer_mut(pointer).unwrap() = value;
        assert!(
            audit::validate(&witness, &x.task, &forged).is_err(),
            "{pointer}"
        );
    }
    let mut tight = witness.clone();
    tight.radius = 2.;
    assert!(audit::validate(&tight, &x.task, &good).is_err());
    Ok(())
}

#[test]
fn integration_orders_by_training_ce_and_retains_the_earlier_choice() -> Result<()> {
    let ideal = ideal_loss(97);
    let modes = ["full_batch", "minibatch", "forward_only"];
    let own_ce = [ideal + 5e-9, ideal + 0.3, ideal + 0.2];
    let prior_ce = [ideal, ideal + 0.2, ideal + 0.3];
    let candidate = |mode: &str, ce: f64, pass: bool| {
        json!({"kind":"paired_fold","task":"t","training_mode":mode,"vocab":97,"train_ce":ce,
            "primary_pass":pass,"all_splits_pass":pass})
    };
    let own = json!({"candidates":(0..3).map(|i| candidate(modes[i], own_ce[i], i == 0))
        .collect::<Vec<_>>()});
    let prior = |mode: &str, ce: f64| {
        json!({"task":"t","prior_training_mode":mode,"candidates":["older", "candidates"],
            "selected":{"kind":"older","vocab":97,"train_ce":ce,"primary_pass":true,
            "all_splits_pass":false}})
    };
    let parent = json!({"choices":(0..3).map(|i| prior(modes[i], prior_ce[i]))
        .collect::<Vec<_>>()});
    let joined = combine(&own, &parent, 1e-8)?;
    assert_eq!(joined["choices_compared"], 3);
    // First within tolerance wins although the earlier one is exactly ideal.
    assert_eq!(joined["choices"][0]["selected"]["kind"], "paired_fold");
    // Outside the tolerance the lower training CE wins, in either direction.
    assert_eq!(joined["choices"][1]["selected"]["kind"], "older");
    assert_eq!(joined["choices"][2]["selected"]["kind"], "paired_fold");
    assert_eq!(joined["paired_selected_choices"], 2);
    // A selected candidate may lose accuracy; nothing held out is consulted.
    assert_eq!(joined["primary_pass_choices"], 2);
    assert_eq!(joined["earlier_primary_pass_choices"], 3);
    assert_eq!(joined["all_split_pass_choices"], 1);
    for (i, choice) in joined["choices"].as_array().unwrap().iter().enumerate() {
        assert_eq!(choice["baseline"], parent["choices"][i]);
        assert_eq!(choice["candidates"][0], own["candidates"][i]);
        assert_eq!(choice["candidates"][1], parent["choices"][i]["selected"]);
    }
    assert_eq!(joined["full35_result"], false);

    let mut repeated = parent.clone();
    repeated["choices"][1] = repeated["choices"][0].clone();
    assert!(combine(&own, &repeated, 1e-8).is_err());
    let mut short = parent.clone();
    short["choices"].as_array_mut().unwrap().pop();
    assert!(combine(&own, &short, 1e-8).is_err());
    let mut unknown = parent.clone();
    unknown["choices"][0]["task"] = "u".into();
    assert!(combine(&own, &unknown, 1e-8).is_err());
    let mut other = parent.clone();
    other["choices"][0]["selected"]["vocab"] = 96.into();
    assert!(combine(&own, &other, 1e-8).is_err());
    let mut below = parent.clone();
    below["choices"][0]["selected"]["train_ce"] = (ideal - 1e-3).into();
    assert!(combine(&own, &below, 1e-8).is_err());
    let mut doubled = own.clone();
    let extra = doubled["candidates"][0].clone();
    doubled["candidates"].as_array_mut().unwrap().push(extra);
    assert!(combine(&doubled, &parent, 1e-8).is_err());
    let root = temporary()?;
    assert!(integrate(&root, &root.join("absent-parent")).is_err());
    fs::remove_dir_all(root)?;
    Ok(())
}

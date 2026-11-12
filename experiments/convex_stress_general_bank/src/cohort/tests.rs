use super::*;
use std::{
    collections::BTreeMap,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

fn example(n: usize, cell: usize, label: usize) -> data::Example {
    data::Example {
        task: "general-cohort-fixture".into(),
        tokens: vec![(cell / n) as i64, (cell % n) as i64],
        targets: vec![-100, label as i64],
        changes: vec![false, true],
        prompt: vec![],
        answer: vec![],
        limit: 0,
        field: None,
        row_index: None,
    }
}
fn fixture(kind: usize) -> Result<Inputs> {
    let (n, labels, missing) = match kind {
        0 => (3, vec![0, 1, 2, 1, 0, 1, 2, 1, 0], Some(5)),
        1 => (2, vec![0, 0, 0, 1], None),
        _ => (
            5,
            vec![
                0, 0, 1, 1, 2, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 2, 1, 1, 0, 0,
            ],
            Some(0),
        ),
    };
    let train = (0..n * n)
        .rev()
        .filter(|i| Some(*i) != missing)
        .map(|i| example(n, i, labels[i]))
        .collect();
    let validation = if let Some(i) = missing {
        vec![example(n, i, labels[i])]
    } else {
        vec![example(n, 0, labels[0])]
    };
    assemble(
        data::Task {
            vocab: n,
            selection: "validation".into(),
            splits: BTreeMap::from([
                ("train".into(), train),
                ("validation".into(), validation),
                (
                    "test".into(),
                    (0..n * n).map(|i| example(n, i, labels[i])).collect(),
                ),
            ]),
        },
        json!({"case":{"batch":3,"observe":3},"data_fingerprints":{"fixture":kind}}),
    )
}
fn temporary() -> Result<std::path::PathBuf> {
    let root = std::env::temp_dir().join(format!(
        "general-bank-cohort-{}-{}",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    ));
    fs::create_dir(&root)?;
    Ok(root)
}
fn planned(arm: &str) -> Result<Case> {
    cases()?
        .into_iter()
        .find(|c| c.arm == arm)
        .ok_or_else(|| "missing case".into())
}
fn inv(c: &Case, kind: usize) -> Result<Value> {
    Ok(
        json!({"source_fingerprint":fingerprint(),"producer_commit":"0000000000000000000000000000000000000000",
    "protocol":protocol()?,"cases":cases()?,"tasks":[task_record(c,&fixture(kind)?)]}),
    )
}

#[test]
fn protocol_keeps_nine_original_tasks_three_modes_and_64_gib_without_loading_data() -> Result<()> {
    assert_eq!(
        protocol()?["shared_fingerprint"],
        convex_stress_relabel::cohort::fingerprint()
    );
    assert_eq!(cases()?.len(), 27);
    assert_eq!(
        cases()?
            .iter()
            .filter(|c| c.task == "arithmetic-abs-diff-blind")
            .count(),
        3
    );
    assert_eq!(protocol()?["address_space_limit_bytes"], RUN_MEMORY_BYTES);
    assert_eq!(protocol()?["explicit_memory_limit_bytes"], RUN_MEMORY_BYTES);
    assert_eq!(protocol()?["decoder_widths"], json!([1, 2, 3, 4]));
    assert!(case("minibatch--unplanned").is_err());
    let root = temporary()?;
    assert!(prepare(&root).is_err());
    let file = root.join("large.json");
    fs::File::create(&file)?.set_len(1025)?;
    assert!(bytes(&file, 1024).is_err());
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn every_mode_replays_original_batches_full_state_and_independent_raw_audit() -> Result<()> {
    let root = temporary()?;
    for arm in ["full_batch", "minibatch", "forward_only"] {
        let c = planned(arm)?;
        let inv = inv(&c, 0)?;
        run::execute(&root, c.clone(), inv.clone(), fixture(0)?, Instant::now())?;
        assert!(run::execute(&root, c.clone(), inv.clone(), fixture(0)?, Instant::now()).is_err());
        let r = read(&root.join(c.id()).join("result.json"))?;
        assert_eq!(r["candidates"].as_array().unwrap().len(), 4);
        assert_eq!(r["covered_cells"], 9);
        assert_eq!(r["selected_decoder"], 0);
        let selected = &r["candidates"][0];
        assert_eq!(selected["updates"], 3);
        assert_eq!(selected["metrics"]["test"]["exact"], 9);
        assert_eq!(selected["stop_reason"], "ideal_ce_excess");
        assert_eq!(
            selected["costs"]["gradients"],
            match arm {
                "full_batch" => 24,
                "minibatch" => 8,
                _ => 0,
            }
        );
        let seal = replay::replay_with(&root, c.clone(), inv.clone(), fixture(0)?)?;
        assert_eq!(seal["candidates"][0]["audit"]["frequencies"], 4);
        assert!(summary::validate_record(&root, &c, &inv).is_err());
        write_new(&root.join(format!("{}.replay.json", c.id())), &seal)?;
        summary::validate_record(&root, &c, &inv)?;
        for k in 1..4 {
            assert_eq!(r["candidates"][k]["status"], "refused");
        }
    }
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn a_larger_fiber_bank_member_and_complex_phase_endpoints_pass_the_raw_audit() -> Result<()> {
    let root = temporary()?;
    let c = planned("forward_only")?;
    let inventory = inv(&c, 2)?;
    run::execute(
        &root,
        c.clone(),
        inventory.clone(),
        fixture(2)?,
        Instant::now(),
    )?;
    let r = read(&root.join(c.id()).join("result.json"))?;
    assert_eq!(r["selected_decoder"], 1);
    assert_eq!(r["covered_cells"], 25);
    assert_eq!(r["candidates"][1]["costs"]["gradients"], 0);
    let seal = replay::replay_with(&root, c.clone(), inventory, fixture(2)?)?;
    assert_eq!(seal["candidates"][1]["audit"]["frequencies"], 8);
    let x = fixture(0)?;
    let found = search::discover(
        &x.problem,
        serde_json::from_value(protocol()?["search"].clone())?,
    )?;
    let verified = found.verified.unwrap();
    let model = Model::anchor(&verified, 0)?;
    let mut projector = general::projection::Projector::new(model.shape)?;
    let metrics = metrics_all(&model, &x.task)?;
    let report = json!({"decoder_index":0,"moment_norms":projector.norms(&model.latent)?,"metrics":metrics,
        "fit_defect":fit_defect(&model,&x.problem),"ideal_ce_excess":number(&metrics["train"],"loss")?-ideal_loss(3)});
    let raw = audit::validate(&model, &verified, &x.task, &report)?;
    assert!(number(&raw, "largest_imaginary_moment_part")? > 0.5);
    let mut bad = model.clone();
    bad.output[0] += 0.1;
    assert!(audit::validate(&bad, &verified, &x.task, &report).is_err());
    bad = model.clone();
    bad.latent[0] += 0.1;
    assert!(audit::validate(&bad, &verified, &x.task, &report).is_err());
    // Every row remains a probability vector with the certified decoded answer,
    // but independent positive phase choices destroy the nuclear bound.
    bad = model.clone();
    bad.latent.fill(0.);
    for cell in 0..9 {
        let y = bad.output[cell * 3..(cell + 1) * 3]
            .iter()
            .position(|&x| x == 1.)
            .unwrap();
        let z = bad.decoder.iter().position(|&c| c == y).unwrap();
        bad.latent[cell * 5 + z] = 1.;
    }
    bad.certified(&verified)?;
    assert!(
        projector
            .norms(&bad.latent)?
            .iter()
            .any(|n| n.nuclear > 3.0001)
    );
    assert!(audit::validate(&bad, &verified, &x.task, &report).is_err());
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn inconsistent_bank_retains_all_four_refusals_and_replays_without_models() -> Result<()> {
    let root = temporary()?;
    let c = planned("full_batch")?;
    let inv = inv(&c, 1)?;
    run::execute(&root, c.clone(), inv.clone(), fixture(1)?, Instant::now())?;
    let r = read(&root.join(c.id()).join("result.json"))?;
    assert_eq!(r["status"], "bank_inconsistent");
    assert!(r["selected_decoder"].is_null());
    assert!(
        r["candidates"]
            .as_array()
            .unwrap()
            .iter()
            .all(|r| r["status"] == "refused")
    );
    let seal = replay::replay_with(&root, c.clone(), inv.clone(), fixture(1)?)?;
    write_new(&root.join(format!("{}.replay.json", c.id())), &seal)?;
    summary::validate_record(&root, &c, &inv)?;
    fs::create_dir(root.join(c.id()).join("d0"))?;
    assert!(replay::replay_with(&root, c.clone(), inv.clone(), fixture(1)?).is_err());
    assert!(summary::validate_record(&root, &c, &inv).is_err());
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn replay_rejects_rehashed_discovery_state_model_trace_and_changed_selection() -> Result<()> {
    let root = temporary()?;
    let c = planned("minibatch")?;
    let inv = inv(&c, 0)?;
    run::execute(&root, c.clone(), inv.clone(), fixture(0)?, Instant::now())?;
    let out = root.join(c.id());
    let original_result = fs::read(out.join("result.json"))?;
    for file in ["discovery", "state", "model"] {
        let path = if file == "discovery" {
            out.join("discovery.json")
        } else {
            out.join(format!("d0/{file}.json"))
        };
        let original = fs::read(&path)?;
        let mut changed: Value = serde_json::from_slice(&original)?;
        match file {
            "discovery" => changed["producer_checks"] = 999.into(),
            "state" => changed["gradient_sum"][0] = 123.into(),
            _ => changed["latent"][0] = 0.123.into(),
        }
        fs::write(&path, serde_json::to_vec(&changed)?)?;
        let mut r: Value = serde_json::from_slice(&original_result)?;
        let digest = hash(&fs::read(&path)?);
        if file == "discovery" {
            r["discovery_sha256"] = digest.into();
        } else {
            r["candidates"][0][format!("{file}_sha256")] = digest.into();
        }
        fs::write(out.join("result.json"), serde_json::to_vec(&r)?)?;
        assert!(
            replay::replay_with(&root, c.clone(), inv.clone(), fixture(0)?).is_err(),
            "{file}"
        );
        fs::write(&path, original)?;
        fs::write(out.join("result.json"), &original_result)?;
    }
    let path = out.join("d0/trace.jsonl");
    let original = fs::read(&path)?;
    let mut rows: Vec<Value> = std::str::from_utf8(&original)?
        .lines()
        .map(serde_json::from_str)
        .collect::<std::result::Result<_, _>>()?;
    rows[0]["train"]["loss"] = 99.into();
    let changed = rows
        .iter()
        .map(Value::to_string)
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";
    fs::write(&path, changed.as_bytes())?;
    let mut r: Value = serde_json::from_slice(&original_result)?;
    r["candidates"][0]["trace_sha256"] = hash(changed.as_bytes()).into();
    fs::write(out.join("result.json"), serde_json::to_vec(&r)?)?;
    assert!(replay::replay_with(&root, c.clone(), inv.clone(), fixture(0)?).is_err());
    fs::write(&path, original)?;
    for key in ["updates", "snapshot_norm_evaluations", "fit_defect"] {
        let mut r: Value = serde_json::from_slice(&original_result)?;
        r["candidates"][0][key] = 999.into();
        fs::write(out.join("result.json"), serde_json::to_vec(&r)?)?;
        assert!(
            replay::replay_with(&root, c.clone(), inv.clone(), fixture(0)?).is_err(),
            "{key}"
        );
    }
    let mut r: Value = serde_json::from_slice(&original_result)?;
    r["selected_decoder"] = 3.into();
    fs::write(out.join("result.json"), serde_json::to_vec(&r)?)?;
    assert!(replay::replay_with(&root, c.clone(), inv.clone(), fixture(0)?).is_err());
    fs::write(out.join("result.json"), original_result)?;
    replay::replay_with(&root, c, inv, fixture(0)?)?;
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn ce_choice_ignores_held_out_scores_and_minibatch_tails_use_original_order() -> Result<()> {
    let ideal = ideal_loss(3);
    let records = vec![
        json!({"status":"refused"}),
        json!({"status":"complete","train_ce":ideal+0.1,"primary_pass":true}),
        json!({"status":"complete","train_ce":ideal,"primary_pass":false}),
    ];
    assert_eq!(choose(&records, 3)?, Some(2));
    let x = fixture(0)?;
    let c = planned("minibatch")?;
    let s = schedule(&c, &x)?;
    let mut order: Vec<_> = (0..s.count).collect();
    let mut rng = Random::new(10_000);
    let mut canonical = x.problem.observations.clone();
    canonical.sort();
    let mut all = vec![];
    for update in 0..s.per_epoch {
        let next = batch(&x, &s, Mode::Mini, update, &mut rng, &mut order);
        assert_eq!(next.len(), if update == 2 { 2 } else { 3 });
        for (&got, &original) in next.iter().zip(&order[update * s.batch..]) {
            assert_eq!(canonical[got], x.problem.observations[original]);
        }
        all.extend(next);
    }
    all.sort();
    assert_eq!(all, (0..8).collect::<Vec<_>>());
    Ok(())
}

#[test]
fn summaries_refuse_missing_process_slots_and_preserve_explicit_process_failures() -> Result<()> {
    let root = temporary()?;
    let mut inventory = json!({"source_fingerprint":fingerprint(),"producer_commit":"0000000000000000000000000000000000000000",
        "protocol":protocol()?,"cases":cases()?,"tasks":[]});
    inventory["tasks"] = serde_json::to_value(
        cases()?
            .iter()
            .filter(|c| c.arm == "full_batch")
            .map(|c| task_record(c, &fixture(0).unwrap()))
            .collect::<Vec<_>>(),
    )?;
    write_new(&root.join("inventory.json"), &inventory)?;
    assert!(summarize(&root).is_err());
    for c in cases()? {
        fs::write(root.join(format!("{}.run.exit", c.id())), "137\n")?;
        fs::write(
            root.join(format!("{}.log", c.id())),
            "fixture process failure\n",
        )?;
    }
    let summary = summarize(&root)?;
    assert_eq!(summary["cases"], 27);
    assert_eq!(summary["planned_decoder_slots"], 108);
    assert_eq!(summary["process_failures"], 27);
    assert_eq!(summary["complete_model_candidates"], 0);
    assert!(
        summary["by_arm"]
            .as_array()
            .unwrap()
            .iter()
            .all(|r| r["cases"] == 9 && r["primary_passes"] == 0)
    );
    fs::remove_dir_all(root)?;
    Ok(())
}

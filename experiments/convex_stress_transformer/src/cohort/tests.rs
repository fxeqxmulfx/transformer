use super::*;
use std::{
    collections::BTreeMap,
    time::{Instant, SystemTime, UNIX_EPOCH},
};
fn temporary() -> Result<std::path::PathBuf> {
    let root = std::env::temp_dir().join(format!(
        "transformer-cohort-{}-{}",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    ));
    fs::create_dir(&root)?;
    Ok(root)
}
fn fixture(c: &Case, success: bool) -> Result<Inputs> {
    let train: Vec<_> = (0..5)
        .map(|k| {
            let n = 3 + k % 2;
            let mut targets = vec![-100; n];
            targets[0] = 0;
            targets[n - 1] = 0;
            data::Example {
                task: "fixture".into(),
                tokens: (0..n).map(|i| ((i + k) % 8) as i64).collect(),
                targets,
                changes: vec![false; n],
                prompt: vec![],
                answer: vec![],
                limit: 0,
                field: None,
                row_index: None,
            }
        })
        .collect();
    let task = data::Task {
        vocab: 8,
        selection: "validation".into(),
        splits: BTreeMap::from([
            ("train".into(), train.clone()),
            ("validation".into(), train[..3].to_vec()),
            ("test".into(), train[3..].to_vec()),
        ]),
    };
    let row = json!({"name":c.task,"tier":"fixture","train_examples":5,"historical_seed1":{"fixture":true},
        "data_fingerprints":{"train":"fixture-training","validation":"fixture-validation","test":"fixture-test"},
        "case":{"width":8,"heads":2,"layers":1,"batch":2,"max_updates":if success{500}else{1},
            "learning_rate":0.03,"warmup":0,"weight_decay":0.,"target":0.99,"metric":"sequence_accuracy"}});
    let mut x = configure(task, row)?;
    if !success {
        let mut initial = engine::Transformer::new(x.dims, c.seed, x.microbatch, x.hyper)?;
        for e in x.task.splits.get_mut("validation").unwrap() {
            for position in [0, e.tokens.len() - 1] {
                let logits = initial.logits(&e.tokens[..=position], &[position])?;
                let best = logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0;
                e.targets[position] = ((best + 1) % 8) as i64;
            }
        }
    }
    Ok(x)
}
fn copy_inputs(x: &Inputs) -> Inputs {
    Inputs {
        task: data::Task {
            vocab: x.task.vocab,
            selection: x.task.selection.clone(),
            splits: x.task.splits.clone(),
        },
        row: x.row.clone(),
        dims: x.dims,
        microbatch: x.microbatch,
        batch: x.batch,
        cap: x.cap,
        hyper: x.hyper,
        buffer_bytes: x.buffer_bytes,
    }
}
fn inv(c: &Case, x: &Inputs) -> Result<Value> {
    Ok(
        json!({"source_fingerprint":fingerprint(),"protocol":protocol()?,"cases":cases()?,
        "producer_commit":"0000000000000000000000000000000000000000","parent":{},"specifications":[specification(c,x)]}),
    )
}
#[test]
fn all_original_tasks_caps_and_frozen_queue_parent_are_retained() -> Result<()> {
    let cases = cases()?;
    let rows = preflight::rows()?;
    assert_eq!(cases.len(), 35);
    assert_eq!(
        cases
            .iter()
            .map(Case::id)
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        35
    );
    for (c, r) in cases.iter().zip(&rows) {
        assert_eq!(c.task, r["name"]);
        assert_eq!(c.seed, 1);
        assert!([20000, 30000, 100000].contains(&r["case"]["max_updates"].as_u64().unwrap()));
    }
    assert_eq!(
        convex_stress_decode::cohort::fingerprint(),
        protocol()?["queue_parent_fingerprint"]
    );
    assert_eq!(protocol()?["seeds"], json!([1]));
    assert!(!observe(249, 100000)?);
    assert!(observe(250, 100000)?);
    assert!(observe(100000, 100000)?);
    let root = temporary()?;
    assert!(prepare(&root.join("new"), &root.join("absent")).is_err());
    assert!(!root.join("new").exists());
    fs::remove_dir_all(root)?;
    Ok(())
}
#[test]
fn cap_failure_replays_and_forged_optimizer_moments_fail_with_updated_digests() -> Result<()> {
    let root = temporary()?;
    let c = cases()?.remove(0);
    let x = fixture(&c, false)?;
    let inv = inv(&c, &x)?;
    write_new(&root.join("inventory.json"), &inv)?;
    run::execute(
        &root,
        c.clone(),
        inv.clone(),
        copy_inputs(&x),
        Instant::now(),
    )?;
    assert!(
        run::execute(
            &root,
            c.clone(),
            inv.clone(),
            copy_inputs(&x),
            Instant::now()
        )
        .is_err()
    );
    let out = root.join(c.id());
    let result_path = out.join("result.json");
    let original_result = fs::read(&result_path)?;
    let r: Value = serde_json::from_slice(&original_result)?;
    assert_eq!(r["stop_reason"], "original_update_cap");
    assert_eq!(r["progress"]["completed_steps"], 1);
    assert_eq!(r["metrics"]["validation"]["sequence_accuracy"], 0.);
    let seal = replay::replay_with(&root, c.clone(), inv.clone(), copy_inputs(&x))?;
    assert_eq!(seal["final_update_bit_exact"], true);
    assert_eq!(seal["observations_replayed"], 2);
    write_new(&root.join(format!("{}.replay.json", c.id())), &seal)?;
    summary::verified_record(&root, &c, &inv)?;

    let endpoint = out.join("at-000001.bin");
    let original_endpoint = fs::read(&endpoint)?;
    let binding = r["metadata"]["binding"].as_str().unwrap();
    let mut state = checkpoint::load(&endpoint, binding)?;
    state.model.m[0] += 0.25;
    let altered = out.join("altered.bin");
    let new_body =
        checkpoint::save_new(&altered, &state.model, &state.rng, state.progress, binding)?;
    fs::rename(altered, &endpoint)?;
    let observation_path = out.join("observations.jsonl");
    let original_observations = fs::read(&observation_path)?;
    let mut observations = vec![];
    replay::lines(&observation_path, |o| {
        observations.push(o);
        Ok(())
    })?;
    let last = observations.last_mut().unwrap();
    last["checkpoint"]["body_sha256"] = new_body.into();
    last["checkpoint"]["file_sha256"] = file_hash(&endpoint)?.into();
    let mut changed = r.clone();
    changed["endpoint"] = last.clone();
    let mut text = String::new();
    for o in observations {
        text.push_str(&serde_json::to_string(&o)?);
        text.push('\n');
    }
    fs::write(&observation_path, text)?;
    changed["observations_sha256"] = file_hash(&observation_path)?.into();
    fs::write(&result_path, serde_json::to_vec(&changed)?)?;
    let error = replay::replay_with(&root, c.clone(), inv.clone(), copy_inputs(&x))
        .unwrap_err()
        .to_string();
    assert!(error.contains("final AdamW update"), "{error}");
    assert!(summary::verified_record(&root, &c, &inv).is_err());
    fs::write(&endpoint, original_endpoint)?;
    fs::write(&observation_path, original_observations)?;
    fs::write(&result_path, original_result)?;
    replay::replay_with(&root, c.clone(), inv.clone(), x)?;
    summary::verified_record(&root, &c, &inv)?;
    fs::remove_dir_all(root)?;
    Ok(())
}
#[test]
fn validation_stopping_uses_the_first_scheduled_success_and_replays_all_splits() -> Result<()> {
    let root = temporary()?;
    let c = cases()?.remove(0);
    let x = fixture(&c, true)?;
    let inv = inv(&c, &x)?;
    run::execute(
        &root,
        c.clone(),
        inv.clone(),
        copy_inputs(&x),
        Instant::now(),
    )?;
    let r = read(&root.join(c.id()).join("result.json"))?;
    assert_eq!(r["stop_reason"], "primary_target");
    assert_eq!(r["progress"]["completed_steps"], 250);
    assert_eq!(r["progress"]["examples_presented"], 500);
    assert_eq!(r["progress"]["targets_presented"], 1000);
    assert_eq!(r["metrics"]["test"]["sequence_accuracy"], 1.);
    let seal = replay::replay_with(&root, c, inv, x)?;
    assert_eq!(seal["original_metrics_replayed"], true);
    assert_eq!(seal["complete_training_rerun"], false);
    fs::remove_dir_all(root)?;
    Ok(())
}

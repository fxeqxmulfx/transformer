use super::*;
use std::{
    collections::BTreeMap,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

fn example(n: usize, cell: usize, label: usize) -> data::Example {
    data::Example {
        task: "certified-fixture".into(),
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
fn fixture(ambiguous: bool) -> Result<Inputs> {
    let (n, labels, missing) = if ambiguous {
        (2, vec![0, 1, 1, 0], 3)
    } else {
        (3, vec![0, 1, 2, 1, 0, 1, 2, 1, 0], 5)
    };
    let train = (0..n * n)
        .rev()
        .filter(|&i| i != missing)
        .map(|i| example(n, i, labels[i]))
        .collect();
    assemble(
        data::Task {
            vocab: n,
            selection: "validation".into(),
            splits: BTreeMap::from([
                ("train".into(), train),
                (
                    "validation".into(),
                    vec![example(n, missing, labels[missing])],
                ),
                (
                    "test".into(),
                    (0..n * n).map(|i| example(n, i, labels[i])).collect(),
                ),
            ]),
        },
        json!({"case":{"batch":3,"observe":3},"data_fingerprints":{"fixture":if ambiguous { "ambiguous-v1" } else { "held-v1" }}}),
    )
}
fn temporary() -> Result<std::path::PathBuf> {
    let root = std::env::temp_dir().join(format!(
        "certified-cohort-{}-{}",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    ));
    fs::create_dir(&root)?;
    Ok(root)
}
fn planned(mode: &str) -> Result<Case> {
    cases()?
        .into_iter()
        .find(|c| c.arm == mode)
        .ok_or_else(|| "missing case".into())
}
fn fixture_inventory(c: &Case, ambiguous: bool) -> Result<Value> {
    Ok(
        json!({"source_fingerprint":fingerprint(),"protocol":protocol()?,"cases":cases()?,
        "producer_commit":"0000000000000000000000000000000000000000","tasks":[task_record(&c.task,&fixture(ambiguous)?)]}),
    )
}

#[test]
fn frozen_protocol_matches_parent_scope_and_new_memory_limit_without_building_original_data()
-> Result<()> {
    let p = protocol()?;
    assert_eq!(
        p["parent_fingerprint"],
        convex_stress_paired::cohort::fingerprint()
    );
    assert_eq!(
        p["tasks"],
        convex_stress_paired::cohort::protocol()?["tasks"]
    );
    assert_eq!(cases()?.len(), 24);
    assert_eq!(p["address_space_limit_bytes"], RUN_MEMORY_BYTES);
    assert_eq!(p["explicit_memory_limit_bytes"], RUN_MEMORY_BYTES);
    assert_eq!(count(&p, "repair_rounds")?, 12);
    let bound = preflight(
        Shape {
            side: 97,
            orbits: 97,
            vocab: 97,
        },
        RUN_MEMORY_BYTES,
    )?;
    assert!(bound < RUN_MEMORY_BYTES);
    assert!(case("minibatch--unplanned-task--fold--rank_one").is_err());
    let root = temporary()?;
    assert!(prepare(&root).is_err());
    let large = root.join("large.json");
    fs::File::create(&large)?.set_len(1025)?;
    assert!(bytes(&large, 1024).is_err());
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn three_modes_replay_every_update_and_raw_audit_the_folded_holdout() -> Result<()> {
    let root = temporary()?;
    for arm in ["full_batch", "minibatch", "forward_only"] {
        let c = planned(arm)?;
        let inv = fixture_inventory(&c, false)?;
        run::execute(
            &root,
            c.clone(),
            inv.clone(),
            fixture(false)?,
            Instant::now(),
        )?;
        assert!(
            run::execute(
                &root,
                c.clone(),
                inv.clone(),
                fixture(false)?,
                Instant::now()
            )
            .is_err()
        );
        let r = read(&root.join(c.id()).join("result.json"))?;
        assert_eq!(r["status"], "complete");
        assert_eq!(r["covered_cells"], 9);
        assert_eq!(r["stop_reason"], "ideal_ce_excess");
        assert_eq!(r["updates"], 3);
        assert_eq!(r["metrics"]["validation"]["sequence_accuracy"], 1.);
        assert_eq!(r["metrics"]["test"]["exact"], 9);
        assert_eq!(
            r["training_gradient_evaluations"],
            match arm {
                "full_batch" => 24,
                "minibatch" => 8,
                _ => 0,
            }
        );
        let discovery = read(&root.join(c.id()).join("discovery.json"))?;
        assert_eq!(
            discovery["attempts"].as_array().unwrap().len() as u64,
            discovery["costs"]["candidates"].as_u64().unwrap()
        );
        let seal = replay::replay_with(&root, c.clone(), inv.clone(), fixture(false)?)?;
        assert_eq!(seal["full_update_replay"], true);
        assert_eq!(seal["audit"]["frequencies"], 4);
        assert!(summary::validate_record(&root, &c, &inv).is_err());
        write_new(&root.join(format!("{}.replay.json", c.id())), &seal)?;
        summary::validate_record(&root, &c, &inv)?;
    }
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn refused_case_is_retained_and_replayed_without_a_candidate() -> Result<()> {
    let root = temporary()?;
    let c = planned("full_batch")?;
    let inv = fixture_inventory(&c, true)?;
    run::execute(
        &root,
        c.clone(),
        inv.clone(),
        fixture(true)?,
        Instant::now(),
    )?;
    let r = read(&root.join(c.id()).join("result.json"))?;
    assert_eq!(r["status"], "uncertified");
    assert_eq!(r["covered_cells"], 0);
    let seal = replay::replay_with(&root, c.clone(), inv.clone(), fixture(true)?)?;
    write_new(&root.join(format!("{}.replay.json", c.id())), &seal)?;
    summary::validate_record(&root, &c, &inv)?;
    write_new(
        &root.join(c.id()).join("model.json"),
        &json!({"invented":true}),
    )?;
    assert!(replay::replay_with(&root, c.clone(), inv.clone(), fixture(true)?).is_err());
    assert!(summary::validate_record(&root, &c, &inv).is_err());
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn replay_rejects_changed_certificate_state_endpoint_trace_and_report() -> Result<()> {
    let root = temporary()?;
    let c = planned("minibatch")?;
    let inv = fixture_inventory(&c, false)?;
    run::execute(
        &root,
        c.clone(),
        inv.clone(),
        fixture(false)?,
        Instant::now(),
    )?;
    let out = root.join(c.id());
    let original_result = fs::read(out.join("result.json"))?;
    for file in ["discovery", "state", "model"] {
        let path = out.join(format!("{file}.json"));
        let original = fs::read(&path)?;
        let mut changed: Value = serde_json::from_slice(&original)?;
        match file {
            "discovery" => changed["verified"]["certificate"]["row"][0] = 1.into(),
            "state" => changed["gradients"][0] = 123.into(),
            _ => changed["latent_half"][0] = 0.into(),
        }
        fs::write(&path, serde_json::to_vec(&changed)?)?;
        let mut r: Value = serde_json::from_slice(&original_result)?;
        r[format!("{file}_sha256")] = hash(&fs::read(&path)?).into();
        fs::write(out.join("result.json"), serde_json::to_vec(&r)?)?;
        assert!(
            replay::replay_with(&root, c.clone(), inv.clone(), fixture(false)?).is_err(),
            "{file}"
        );
        fs::write(&path, original)?;
        fs::write(out.join("result.json"), &original_result)?;
    }
    let trace_path = out.join("trace.jsonl");
    let original_trace = fs::read(&trace_path)?;
    let mut rows: Vec<Value> = std::str::from_utf8(&original_trace)?
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
    fs::write(&trace_path, changed.as_bytes())?;
    let mut r: Value = serde_json::from_slice(&original_result)?;
    r["trace_sha256"] = hash(changed.as_bytes()).into();
    fs::write(out.join("result.json"), serde_json::to_vec(&r)?)?;
    assert!(replay::replay_with(&root, c.clone(), inv.clone(), fixture(false)?).is_err());
    fs::write(&trace_path, original_trace)?;
    for key in [
        "updates",
        "covered_cells",
        "snapshot_norm_evaluations",
        "full_operator_residual",
        "fit_defect",
    ] {
        let mut r: Value = serde_json::from_slice(&original_result)?;
        r[key] = 999.into();
        fs::write(out.join("result.json"), serde_json::to_vec(&r)?)?;
        assert!(
            replay::replay_with(&root, c.clone(), inv.clone(), fixture(false)?).is_err(),
            "{key}"
        );
    }
    fs::write(out.join("result.json"), &original_result)?;
    replay::replay_with(&root, c, inv, fixture(false)?)?;
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn selection_uses_training_ce_and_keeps_refusals_and_earlier_choices() -> Result<()> {
    let ideal = ideal_loss(3);
    let old = json!({"vocab":3,"train_ce":ideal+0.1,"all_splits_pass":true,"primary_pass":true});
    let parent = json!({"choices":[{"task":"a","training_mode":"minibatch","selected":old},
        {"task":"b","training_mode":"minibatch","selected":old}]});
    let own = json!({"records":[{"task":"b","status":"uncertified"}],"candidates":[
        {"task":"a","training_mode":"minibatch","vocab":3,"train_ce":ideal,
         "primary_pass":false,"all_splits_pass":false}]});
    let result = integrate::combine(&own, &parent, 1e-8)?;
    assert_eq!(result["certified_selected_choices"], 1);
    assert_eq!(result["choices"][0]["selected"]["all_splits_pass"], false);
    assert_eq!(result["choices"][1]["selected"], old);
    assert_eq!(result["choices"][0]["baseline"], parent["choices"][0]);
    assert_eq!(result["refused_or_completed_records"], own["records"]);
    Ok(())
}

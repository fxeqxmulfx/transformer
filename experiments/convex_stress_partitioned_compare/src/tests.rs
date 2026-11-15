use super::*;

#[test]
fn actual_complete_models_preserve_fallback_and_replay_all_logits() -> Result<()> {
    let (parent, task, _) = fixture::fixture(old::Mode::Forward)?;
    let sha = fresh::digest(&parent)?;
    let input = convex_stress_shared_head::inspection::extract(&parent, "prior", &sha)?;
    let coverage = model::coverage(&input)?;
    assert!(input.records.iter().all(|r| coverage[input.route[r.cell]]));
    let outcome = fresh::training::run(input.clone(), policy("forward_only")?)?;
    assert_eq!(outcome.work.gradient_batches, 0);
    assert_eq!(outcome.work.gradient_records, 0);
    let saved = model::Saved {
        parent_sha256: sha,
        prior: outcome,
        supplement: None,
    };
    let loaded: model::Saved = serde_json::from_slice(&serde_json::to_vec(&saved)?)?;
    let runtime = model::Model::load(parent.clone(), loaded)?;
    let (mut fallback, mut covered) = (0, 0);
    for cell in 0..49 {
        let new = runtime.logits(cell)?;
        let direct = runtime.independent(cell)?;
        assert!(new.iter().zip(direct).all(|(a, b)| (a - b).abs() < 1e-9));
        if runtime.routed(cell).0.covered(cell) {
            covered += 1;
            assert_eq!(new, runtime.prior.outcome.model.logits_cell(cell)?);
        } else {
            fallback += 1;
            assert_eq!(new, runtime.old_logits(cell)?);
            assert!(new.iter().any(|&x| x != 0.));
        }
    }
    assert!(covered > 0);
    assert!(fallback > 0);
    let folder = std::env::temp_dir().join(format!("partitioned-adapter-{}", std::process::id()));
    fs::create_dir(&folder)?;
    let path = folder.join("validation.jsonl");
    let a = audit::run(&runtime, &task.splits["validation"], &path, false, None)?;
    let b = audit::run(&runtime, &task.splits["validation"], &path, true, None)?;
    assert_eq!(a, b);
    assert_eq!(a["targets"], 49);
    let lines = fs::read_to_string(&path)?;
    assert_eq!(lines.lines().count(), 49);
    let first: Value = serde_json::from_str(lines.lines().next().unwrap())?;
    assert_eq!(first["old_logits"].as_array().unwrap().len(), 7);
    assert_eq!(first["new_logits"].as_array().unwrap().len(), 7);
    let reference = folder.join("reference.jsonl");
    let mut reference_file = fs::File::create(&reference)?;
    for line in lines.lines() {
        let mut row: Value = serde_json::from_str(line)?;
        row["retained330"] = json!({"logits":row["old_logits"],
            "prediction":row["old_prediction"],"ce":row["old_ce"]});
        serde_json::to_writer(&mut reference_file, &row)?;
        reference_file.write_all(b"\n")?;
    }
    drop(reference_file);
    let joined = folder.join("joined.jsonl");
    let joined_audit = audit::run(
        &runtime,
        &task.splits["validation"],
        &joined,
        false,
        Some(&reference),
    )?;
    assert_eq!(joined_audit["retained332"]["correct"], a["correct"]);
    assert_eq!(joined_audit["retained332"]["fixed"], 0);
    assert_eq!(joined_audit["retained332"]["regressed"], 0);
    assert_eq!(joined_audit["retained330"]["correct"], a["parent_correct"]);
    assert_eq!(joined_audit["retained330"]["fixed"], a["fixed"]);
    assert_eq!(joined_audit["retained330"]["regressed"], a["regressed"]);
    assert_eq!(
        joined_audit,
        audit::run(
            &runtime,
            &task.splits["validation"],
            &joined,
            true,
            Some(&reference)
        )?
    );
    let mut corrupted = first;
    corrupted["target"] = json!(99);
    let wrong = folder.join("wrong-reference.jsonl");
    fs::write(&wrong, format!("{}\n", serde_json::to_string(&corrupted)?))?;
    assert!(
        audit::run(
            &runtime,
            &task.splits["validation"],
            &folder.join("refused.jsonl"),
            false,
            Some(&wrong)
        )
        .is_err()
    );
    fs::remove_dir_all(folder)?;
    let mut bad = saved;
    bad.prior.model.theta[0] += 0.5;
    assert!(model::Model::load(parent, bad).is_err());
    Ok(())
}
#[test]
fn forecasts_cover_field_calls_without_heldout_label_selection() -> Result<()> {
    let (parent, mut task, _) = fixture::fixture(old::Mode::Forward)?;
    let sha = fresh::digest(&parent)?;
    let before = costs::forecast(&parent, &sha, &task, "forward_only", 1000, 1000)?;
    assert_eq!(before["calls"]["old_metric_calls"], 94);
    assert_eq!(before["calls"]["new_raw_audit_calls"], 94);
    for e in task.splits.get_mut("validation").unwrap() {
        e.targets[1] = (e.targets[1] + 1) % 7;
    }
    assert_eq!(
        before,
        costs::forecast(&parent, &sha, &task, "forward_only", 1000, 1000)?
    );
    for e in task.splits.get_mut("validation").unwrap() {
        e.field = Some((0, 0));
    }
    let field = costs::forecast(&parent, &sha, &task, "forward_only", 1000, 1000)?;
    assert_eq!(field["calls"]["old_metric_calls"], 143);
    assert_eq!(field["calls"]["new_metric_calls"], 143);
    assert!(field["total_kernel_work"].as_u64() > before["total_kernel_work"].as_u64());
    task.splits.get_mut("train").unwrap()[0].targets[1] = 6;
    assert!(costs::forecast(&parent, &sha, &task, "forward_only", 1000, 1000).is_err());
    assert!(costs::forecast(&parent, &sha, &task, "forward_only", 1 << 31, 1000).is_err());
    Ok(())
}
#[test]
fn gate_uses_training_coverage_and_preserves_original_role_route() -> Result<()> {
    for prior in [false, true] {
        for supplement in [None, Some(false), Some(true)] {
            assert_eq!(
                model::use_supplement(prior, supplement),
                !prior && supplement == Some(true)
            );
        }
    }
    let mut input = fresh::Input {
        parent_sha256: "a".repeat(64),
        source_sha256: "b".repeat(64),
        n: 2,
        vocabulary: 2,
        blocks: 2,
        route: vec![0, 1, 0, 1],
        records: vec![
            fresh::Record {
                id: 0,
                cell: 0,
                label: 0,
            },
            fresh::Record {
                id: 1,
                cell: 0,
                label: 0,
            },
        ],
    };
    assert_eq!(model::coverage(&input)?, vec![true, false]);
    for r in &mut input.records {
        r.label = 1;
    }
    assert_eq!(model::coverage(&input)?, vec![true, false]);
    input.records.push(fresh::Record {
        id: 2,
        cell: 1,
        label: 0,
    });
    assert_eq!(model::coverage(&input)?, vec![true, true]);
    input.records[1].id = 0;
    assert!(model::coverage(&input).is_err());
    Ok(())
}
#[test]
fn both_complete_roles_and_every_training_mode_retain_the_frozen_route() -> Result<()> {
    for (mode, label) in [
        (old::Mode::Full, "full_batch"),
        (old::Mode::Mini, "minibatch"),
        (old::Mode::Forward, "forward_only"),
    ] {
        let (a, _, _) = fixture::with_prime(mode, 3)?;
        let (b, _, _) = fixture::with_prime(mode, 2)?;
        let parent = json!({"prior":a["prior"],"supplement":b["prior"]});
        let sha = fresh::digest(&parent)?;
        let train = |role: &str| -> Result<fresh::training::Outcome> {
            fresh::training::run(
                convex_stress_shared_head::inspection::extract(&parent, role, &sha)?,
                policy(label)?,
            )
        };
        let saved = model::Saved {
            parent_sha256: sha.clone(),
            prior: train("prior")?,
            supplement: Some(train("supplement")?),
        };
        let m = model::Model::load(parent, saved)?;
        let supplement = m.supplement.as_ref().unwrap();
        let mut routes = 0;
        for cell in 0..49 {
            let use_second =
                !m.prior.old.source().known(cell) && supplement.old.source().known(cell);
            let expected = if use_second { supplement } else { &m.prior };
            routes += usize::from(use_second);
            assert_eq!(
                m.routed(cell).1,
                if use_second { "supplement" } else { "prior" }
            );
            assert_eq!(m.old_logits(cell)?, expected.old.logits_cell(cell)?);
            let prediction = if expected.covered(cell) {
                expected.outcome.model.logits_cell(cell)?
            } else {
                expected.old.logits_cell(cell)?
            };
            assert_eq!(m.logits(cell)?, prediction);
        }
        assert!(routes > 0);
        assert_eq!(
            m.info()["complete_real_coefficient_slots"],
            m.prior.old.source().coefficients.len()
                + m.prior.outcome.model.input.blocks
                    * (1 + 2 * m.prior.outcome.model.bank.states())
                    * m.prior.outcome.model.input.vocabulary
                + supplement.old.source().coefficients.len()
                + supplement.outcome.model.input.blocks
                    * (1 + 2 * supplement.outcome.model.bank.states())
                    * supplement.outcome.model.input.vocabulary
        );
    }
    Ok(())
}
fn baseline() -> Value {
    let m = json!({"targets":100,"examples":100,"correct":100,"native_correct":100,"native_token_accuracy":1.,"token_accuracy":1.,"sequence_accuracy":1.,"loss":0.01});
    let choices:Vec<_>=(0..35).flat_map(|i|["full_batch","minibatch","forward_only"].map({let m=m.clone();move|mode|json!({
        "task":format!("task{i}"),"reference":format!("fixture:{i}:{mode}"),"training_mode":mode,"primary_split":"validation",
        "primary_pass":true,"all_splits_pass":true,"train_ce":0.01,"metrics":{"train":m,"validation":m}})})).collect();
    let arms = json!([{"initializer":"minimum_ce","choices":choices},{"initializer":"first_small_ce","choices":choices}]);
    let records: Vec<_> = (0..35)
        .map(|i| json!({"task":format!("task{i}"),"status":"complete"}))
        .collect();
    let r = json!({"arms":arms});
    json!({"tasks":35,"choices_per_arm":105,"process_failures":0,"arms":arms,"records":records,
        "baselines":{"transformer_endpoints":vec![0;35],"depth_outcomes":vec![0;336]},
        "prior_crossing_reference":r,"prior_exclusion_reference":r,"prior_region_reference":r,"prior_local_reference":r})
}
#[test]
fn changed_raw_and_ce_regressions_remain_in_both_complete_policies() -> Result<()> {
    let parent = baseline();
    let mut results: Vec<_> = (0..35)
        .map(|i| json!({"task":format!("task{i}"),"status":"complete","outcomes":[]}))
        .collect();
    let mut new = parent["arms"][0]["choices"][0].clone();
    new["parent_reference"] = new["reference"].clone();
    new["reference"] = json!("new-state");
    new["train_ce"] = json!(0.2);
    new["metrics"]["validation"]["native_correct"] = json!(98);
    new["metrics"]["validation"]["native_token_accuracy"] = json!(0.98);
    new["metrics"]["validation"]["loss"] = json!(0.3);
    new["model_info"] = json!({"new_optimizer_updates":1,"new_gradient_batches":1});
    new["metric_provenance"] = json!({"kind":"fresh_changed_model"});
    results[0]["outcomes"] = json!([new]);
    let mut combined = summary::combine(parent.clone(), &results)?;
    for arm in combined["per_task99"].as_array().unwrap() {
        let full = &arm["modes"][0];
        assert_eq!(full["cases"], 35);
        assert_eq!(full["primary_raw99"], 34);
        assert_eq!(full["original_successes"], 35);
    }
    assert_eq!(combined["arms"][0]["choices"][0]["train_ce"], 0.2);
    assert_eq!(combined["baselines"], parent["baselines"]);
    assert_eq!(combined["retained319"]["arms"], parent["arms"]);
    assert_eq!(combined["changes_vs319"].as_array().unwrap().len(), 210);
    let mut previous = parent.clone();
    for arm in previous["arms"].as_array_mut().unwrap() {
        arm["choices"][0]["metrics"]["validation"]["native_correct"] = json!(97);
        arm["choices"][0]["metrics"]["validation"]["native_token_accuracy"] = json!(0.97);
    }
    summary::retain_comparison(&mut combined, &previous)?;
    summary::retain_330(&mut combined, &parent)?;
    assert_eq!(combined["changes_vs330"].as_array().unwrap().len(), 210);
    assert_eq!(combined["changes_vs330"][0]["before_raw_correct"], 100);
    assert_eq!(combined["changes_vs332"].as_array().unwrap().len(), 210);
    assert_eq!(combined["changes_vs332"][0]["before_raw_correct"], 97);
    assert_eq!(combined["changes_vs332"][0]["after_raw_correct"], 98);
    assert_eq!(combined["changes_vs319"][0]["before_raw_correct"], 100);
    previous["arms"][0]["choices"][0]["metrics"]["validation"]["targets"] = json!(99);
    assert!(summary::retain_comparison(&mut combined, &previous).is_err());
    results[0]["status"] = json!("failed");
    assert!(summary::combine(parent.clone(), &results).is_err());
    results.pop();
    assert!(summary::combine(parent, &results).is_err());
    Ok(())
}

#[test]
fn phase_memory_refuses_oversize_and_keeps_sequential_peak() -> Result<()> {
    let p = costs::memory(
        350_000_000,
        330_000_000,
        290_000_000,
        22_600_000_000,
        13_500_000_000,
    )?;
    assert!(p["maximum"].as_u64().unwrap() < BUFFER_LIMIT);
    assert_eq!(p["maximum"], p["load_and_score"]);
    assert!(
        costs::memory(
            350_000_000,
            330_000_000,
            400_000_000,
            22_600_000_000,
            13_500_000_000
        )
        .is_err()
    );
    assert!(costs::memory(u64::MAX, 0, 0, 0, 0).is_err());
    assert!(costs::memory(0, 0, 0, BUFFER_LIMIT, 1).is_err());
    assert!(costs::saved_bytes("unknown").is_err());
    Ok(())
}

use super::*;
mod fixtures;
mod runner;
use fixtures::*;
#[test]
fn full_catalogue_replays_shared_role_models_and_keeps_failed_scope() -> Result<()> {
    let f = Fixture::new()?;
    f.close()?;
    let summary = f.summary()?;
    assert_eq!(summary["tasks"].as_array().unwrap().len(), 35);
    let c = &summary["constructor_accounting"];
    assert_eq!(c["planned_cases"], 2);
    assert_eq!(c["completed_cases"], 1);
    assert_eq!(c["process_failures"], 1);
    assert_eq!(c["complete_stages"], 55);
    assert_eq!(c["available_endpoints_by_mode"], json!([1, 1, 1]));
    assert_eq!(c["complete_gradient_batches"], 5);
    assert_eq!(c["complete_restricted_kernels"], 1);
    assert_eq!(c["admitted_raw_expansions"], 1);
    assert_eq!(summary["declared_readout_used_in_discovery"], true);
    assert_eq!(summary["original_raw_ce_unchanged"], true);
    assert_eq!(summary["original_record_multiplicity_retained"], true);
    assert_eq!(summary["shared_role_target_proved"], false);
    assert_eq!(c["partial_costs_in_retained_resources"], true);
    assert!(
        !summary["records"][1]["artifacts"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    let a: stage::Attempt = read(&f.root.join(&f.planned[0]).join("stage-03-p7.json"))?;
    let d = a.discovery.as_ref().unwrap();
    let h = a.head.as_ref().unwrap();
    assert_eq!(h.original_records, 7);
    assert_eq!(d.request.input.observations.len(), 5);
    assert_eq!(
        h.endpoints[1]
            .training
            .as_ref()
            .unwrap()
            .steps
            .last()
            .unwrap()
            .count,
        1
    );
    assert_eq!(d.raw_code.as_ref().unwrap().dimension, 1);
    assert_ne!(
        d.raw_code.as_ref().unwrap().label(0),
        d.raw_code.as_ref().unwrap().label(2)
    );
    assert!(
        a.head
            .as_ref()
            .unwrap()
            .endpoints
            .iter()
            .all(|e| e.audit.as_ref().unwrap().record_errors == 0)
    );
    Ok(())
}
#[test]
fn original_scope_source_binary_request_and_protocol_mutations_fail() -> Result<()> {
    let f = Fixture::new()?;
    let path = f.root.join("inventory.json");
    let original: Value = read(&path)?;
    for kind in 0..16 {
        let mut v = original.clone();
        match kind {
            0 => {
                v["tasks"].as_array_mut().unwrap().pop();
            }
            1 => {
                v["planned"].as_array_mut().unwrap().pop();
            }
            2 => v["primes"].as_array_mut().unwrap().reverse(),
            3 => v["source_fingerprint"] = json!("changed"),
            4 => v["binary_sha256"] = json!("changed"),
            5 => v["request"]["producer_commit"] = json!("b".repeat(40)),
            6 => v["protocol"]["memory_limit_bytes"] = json!(1),
            7 => v["heldout_evaluation"] = json!(true),
            8 => v["parent_comparison"] = json!(true),
            9 => v["raw_label_identification"] = json!(true),
            10 => v["inputs"].as_array_mut().unwrap().reverse(),
            11 => v["declared_readout_used_in_discovery"] = json!(false),
            12 => v["original_raw_ce_unchanged"] = json!(false),
            13 => v["shared_role_target_proved"] = json!(true),
            14 => v["shared_role_constraints_used"] = json!(false),
            _ => v["original_record_multiplicity_retained"] = json!(false),
        }
        overwrite(&path, &v);
        assert!(f.inventory().is_err(), "mutation{kind}");
    }
    overwrite(&path, &original);
    let mut request: Request = read(&f.root.join("request.json"))?;
    request.suite_sha256 = "changed".into();
    overwrite(&f.root.join("request.json"), &request);
    assert!(f.inventory().is_err());
    Ok(())
}
#[test]
fn original_labels_order_batch_and_heldout_independence_are_bound() -> Result<()> {
    let f = Fixture::new()?;
    f.run_first()?;
    let inv = f.inventory()?;
    for kind in 0..9 {
        let mut x = input(&f.planned[0])?;
        match kind {
            0 => x.input.observations.reverse(),
            1 => x.input.observations[0][1] = 6,
            2 => x.policies[1].batch += 1,
            3 => x.input.policy.equations -= 1,
            4 => x.readout.swap(0, 1),
            5 => x.answers = 3,
            6 => x.records.reverse(),
            7 => {
                x.records.pop();
            }
            _ => x.records[0].id = 100,
        }
        assert!(stage::replay_with(&f.root, &f.planned[0], &inv, x).is_err());
    }
    let a = input("name-a")?;
    let mut changed = task();
    changed.splits.get_mut("validation").unwrap()[0].targets[1] = 6;
    let b = inputs::assemble(changed, "name-b", 2, json!({"synthetic":true}))?;
    assert_eq!(a.input, b.input);
    assert_eq!(a.records, b.records);
    assert_eq!(a.policies, b.policies);
    assert_eq!(a.readout, b.readout);
    assert_eq!(a.answers, b.answers);
    for prime in [2, 7] {
        let d = discovery::run(a.request(prime))?;
        assert_eq!(
            binding::run(&d, &a.records, a.policies.clone())?,
            binding::run(&d, &b.records, b.policies.clone())?
        );
    }
    Ok(())
}
#[test]
fn numerical_mutations_fail_after_rehashing_the_entire_retained_ledger() -> Result<()> {
    let f = Fixture::new()?;
    f.run_first()?;
    let dir = f.root.join(&f.planned[0]);
    let path = dir.join("stage-03-p7.json");
    let original: stage::Attempt = read(&path)?;
    let original_result: Value = read(&dir.join("result.json"))?;
    for kind in 0..8 {
        let mut a = original.clone();
        match kind {
            0 => {
                a.discovery
                    .as_mut()
                    .unwrap()
                    .field
                    .as_mut()
                    .unwrap()
                    .space
                    .as_mut()
                    .unwrap()
                    .directions
                    .pop();
            }
            1 => {
                a.head
                    .as_mut()
                    .unwrap()
                    .initialization
                    .as_mut()
                    .unwrap()
                    .model
                    .encoding
                    .unknown_axis_cells += 1
            }
            2 => {
                a.head.as_mut().unwrap().endpoints[0]
                    .training
                    .as_mut()
                    .unwrap()
                    .model
                    .coefficients[0] += 0.2
            }
            3 => {
                a.head.as_mut().unwrap().endpoints[1]
                    .training
                    .as_mut()
                    .unwrap()
                    .steps[0]
                    .head_gradient_l2 += 0.3
            }
            4 => {
                a.discovery
                    .as_mut()
                    .unwrap()
                    .target_readout_compatibility_proved = true
            }
            5 => a.head = None,
            6 => a.discovery_seconds = -1.,
            _ => {
                a.head.as_mut().unwrap().endpoints[2]
                    .training
                    .as_mut()
                    .unwrap()
                    .work
                    .gradient_batches = 1
            }
        }
        overwrite(&path, &a);
        let mut result = original_result.clone();
        result["stages"][3] = json!({"prime":7,"file":"stage-03-p7.json","sha256":digest(&path)?,
            "bytes":fs::metadata(&path)?.len(),"retained":stage::retained(&a)?});
        overwrite(&dir.join("result.json"), &result);
        assert!(
            stage::record(
                &f.root,
                &f.planned[0],
                &f.inventory()?,
                &input(&f.planned[0])?,
                false
            )
            .is_ok()
        );
        assert!(f.replay_first().is_err(), "numerical mutation{kind}");
    }
    overwrite(&path, &original);
    overwrite(&dir.join("result.json"), &original_result);
    f.replay_first()?;
    Ok(())
}
#[test]
fn equation_refusals_keep_every_prime_three_modes_and_zero_gradient_work() -> Result<()> {
    let f = Fixture::new()?;
    let mut x = input(&f.planned[0])?;
    x.input.policy.equations = 2;
    x.record["input_sha256"] = json!(hash(&serde_json::to_vec(&x.input)?));
    let mut inv = f.inventory()?;
    inv["inputs"][0] = x.record.clone();
    stage::run_with(&f.root, &f.planned[0], &inv, x)?;
    let mut x = input(&f.planned[0])?;
    x.input.policy.equations = 2;
    x.record["input_sha256"] = json!(hash(&serde_json::to_vec(&x.input)?));
    stage::replay_with(&f.root, &f.planned[0], &inv, x)?;
    let result: Value = read(&f.root.join(&f.planned[0]).join("result.json"))?;
    assert_eq!(result["stages"].as_array().unwrap().len(), 55);
    for stage in result["stages"].as_array().unwrap() {
        assert_eq!(
            stage["retained"]["discovery"]["reason"],
            if stage["prime"] == 7 {
                "all_role_equation_cap"
            } else {
                "answer_code_cardinality_mismatch"
            }
        );
        assert_eq!(stage["retained"]["discovery"]["work"]["ce_gradients"], 0);
        assert!(
            stage["retained"]["endpoints"]
                .as_array()
                .unwrap()
                .iter()
                .all(|e| e["training"].is_null() && e["refusal"].is_string())
        );
    }
    Ok(())
}
#[test]
fn missing_stage_forged_replay_and_replay_process_failure_stay_explicit() -> Result<()> {
    let f = Fixture::new()?;
    f.close()?;
    let mut inv = f.inventory()?;
    inv["inputs"][1]["input_sha256"] = json!("changed failed-case input");
    assert!(summary::case_record(&f.root, &f.planned[1], &inv, input(&f.planned[1])?).is_err());
    let replay = f.root.join(format!("{}.replay.json", f.planned[0]));
    let original: Value = read(&replay)?;
    let mut changed = original.clone();
    changed["all_complete_restricted_bases_replayed"] = json!(false);
    overwrite(&replay, &changed);
    assert!(f.summary().is_err());
    overwrite(&replay, &original);
    fs::write(f.root.join(format!("{}.replay.exit", f.planned[0])), "2\n")?;
    let summary = f.summary()?;
    assert_eq!(summary["constructor_accounting"]["process_failures"], 2);
    assert_eq!(
        summary["constructor_accounting"]["available_endpoints_by_mode"],
        json!([0, 0, 0])
    );
    fs::write(f.root.join(format!("{}.replay.exit", f.planned[0])), "0\n")?;
    let path = f.root.join(&f.planned[0]).join("result.json");
    let mut result: Value = read(&path)?;
    result["stages"].as_array_mut().unwrap().pop();
    overwrite(&path, &result);
    assert!(f.replay_first().is_err());
    Ok(())
}
#[test]
fn original_contract_matches_reviewed_core_and_complete_task_scope() -> Result<()> {
    let p = protocol()?;
    assert_eq!(tasks(&suite()?)?.len(), 35);
    assert_eq!(planned()?.len(), 23);
    assert_eq!(catalogue().len(), 55);
    assert_eq!(p["required_done_tasks"], json!([271, 272, 273]));
    assert_eq!(p["core_commit"], "5d4eee848cacceba911f048ade2d014e2b71cf42");
    assert_eq!(p["memory_limit_bytes"], 68719476736_u64);
    assert_eq!(kernel::discovery::Policy::default().equations, 8190);
    assert_eq!(kernel::discovery::Policy::default().elimination.nullity, 8);
    for name in planned()? {
        assert!(tasks(&suite()?)?.contains(&name));
    }
    assert!(name("../outside").is_err());
    Ok(())
}

#[test]
fn declared_field_partition_binds_raw_aliases_without_grouping_the_ce_loss() -> Result<()> {
    let mut t = task();
    t.vocab = 197;
    let rows = t.splits.get_mut("train").unwrap();
    rows.truncate(3);
    for (row, label) in rows.iter_mut().zip([98, 99, 2]) {
        row.targets[1] = label;
        row.field = Some((0, (label - 98).rem_euclid(97)));
    }
    let x = inputs::assemble(t, "any-task-name", 2, json!({"synthetic":true}))?;
    assert_eq!(x.answers, 97);
    assert_eq!(x.readout.len(), 197);
    assert_eq!(x.readout[99], x.readout[2]);
    assert_eq!(x.record["readout_changes_features"], true);
    assert_eq!(x.record["readout_changes_ce"], false);
    let d = discovery::run(x.request(97))?;
    discovery::replay(&x.request(97), &d)?;
    assert_eq!(
        d.raw_code.as_ref().unwrap().label(99),
        d.raw_code.as_ref().unwrap().label(2)
    );
    let head = binding::run(&d, &x.records, x.policies.clone())?;
    binding::replay(&d, &x.records, &x.policies, &head)?;
    assert_eq!(head.initialization.as_ref().unwrap().model.vocabulary, 197);
    for e in &head.endpoints {
        assert_eq!(e.audit.as_ref().unwrap().record_errors, 0);
        assert_eq!(
            e.training
                .as_ref()
                .unwrap()
                .model
                .encode(&x.input.observations)?
                .iter()
                .map(|e| e.label)
                .collect::<Vec<_>>(),
            vec![98, 99, 2]
        );
    }
    let mut mixed = task();
    mixed.splits.get_mut("train").unwrap()[0].field = Some((0, 0));
    assert!(inputs::assemble(mixed, "mixed", 2, json!({})).is_err());
    Ok(())
}

#[test]
fn declared_partition_file_is_checked_even_after_its_ledger_hash_is_rebuilt() -> Result<()> {
    let f = Fixture::new()?;
    f.run_first()?;
    let dir = f.root.join(&f.planned[0]);
    let path = dir.join("readout.json");
    let mut q: Value = read(&path)?;
    q["classes"][0] = json!(1);
    overwrite(&path, &q);
    let mut result: Value = read(&dir.join("result.json"))?;
    result["readout_sha256"] = json!(digest(&path)?);
    overwrite(&dir.join("result.json"), &result);
    assert!(f.replay_first().is_err());
    Ok(())
}

#[test]
fn original_repeated_records_are_not_reweighted_or_silently_dropped() -> Result<()> {
    let x = input("records")?;
    assert_eq!(x.records.len(), 7);
    assert_eq!(x.input.observations.len(), 5);
    assert_eq!(
        x.records.iter().map(|r| r.id).collect::<Vec<_>>(),
        (0..7).collect::<Vec<_>>()
    );
    assert_eq!(x.records[1].cell, 1);
    assert_eq!(x.records[5].cell, x.records[0].cell);
    assert_eq!(x.policies[1].updates, 4);
    let mut reduced = task();
    reduced.splits.get_mut("train").unwrap().truncate(5);
    let y = inputs::assemble(reduced, "records", 2, json!({"synthetic":true}))?;
    assert_eq!(x.input, y.input);
    assert_ne!(x.record["records_sha256"], y.record["records_sha256"]);
    assert_ne!(x.policies[1].updates, y.policies[1].updates);
    let mut conflict = task();
    conflict.splits.get_mut("train").unwrap()[5].targets[1] = 2;
    assert!(inputs::assemble(conflict, "conflict", 2, json!({})).is_err());
    let mut unsupported = task();
    unsupported.splits.get_mut("train").unwrap()[0]
        .tokens
        .push(0);
    assert!(inputs::assemble(unsupported, "unsupported", 2, json!({})).is_err());
    Ok(())
}

#[test]
fn original_record_file_tampering_fails_even_after_rehashing() -> Result<()> {
    let f = Fixture::new()?;
    f.run_first()?;
    let dir = f.root.join(&f.planned[0]);
    let path = dir.join("records.json");
    let saved: Vec<records::Record> = read(&path)?;
    let result_path = dir.join("result.json");
    let result: Value = read(&result_path)?;
    for kind in 0..4 {
        let mut changed = saved.clone();
        match kind {
            0 => {
                changed.pop();
            }
            1 => changed[0].id = 100,
            2 => changed.swap(0, 1),
            _ => changed[5].cell = changed[2].cell,
        }
        overwrite(&path, &changed);
        let mut rebuilt = result.clone();
        rebuilt["records_sha256"] = json!(digest(&path)?);
        overwrite(&result_path, &rebuilt);
        assert!(f.replay_first().is_err());
    }
    overwrite(&path, &saved);
    overwrite(&result_path, &result);
    f.replay_first()?;
    Ok(())
}

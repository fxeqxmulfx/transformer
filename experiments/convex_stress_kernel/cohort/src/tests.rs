use super::*;
mod fixtures;
use fixtures::*;
#[test]
fn full_catalogue_replays_multidimensional_models_and_keeps_failed_scope() -> Result<()> {
    let f = Fixture::new()?;
    f.close()?;
    let summary = f.summary()?;
    assert_eq!(summary["tasks"].as_array().unwrap().len(), 35);
    let c = &summary["constructor_accounting"];
    assert_eq!(c["planned_cases"], 2);
    assert_eq!(c["completed_cases"], 1);
    assert_eq!(c["process_failures"], 1);
    assert_eq!(c["complete_stages"], 55);
    assert_eq!(c["available_endpoints_by_mode"], json!([55, 55, 55]));
    assert_eq!(c["complete_gradient_batches"], 220);
    assert_eq!(c["partial_costs_in_retained_resources"], true);
    assert!(
        !summary["records"][1]["artifacts"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    let a: stage::Attempt = read(&f.root.join(&f.planned[0]).join("stage-00-p2.json"))?;
    let d = a.discovery.as_ref().unwrap();
    assert_eq!(d.code.as_ref().unwrap().dimension, 2);
    assert_eq!(
        d.code.as_ref().unwrap().label(0),
        d.code.as_ref().unwrap().label(2)
    );
    assert!(
        a.head
            .as_ref()
            .unwrap()
            .endpoints
            .iter()
            .all(|e| e.audit.as_ref().unwrap().train_errors == 0)
    );
    Ok(())
}
#[test]
fn original_scope_source_binary_request_and_protocol_mutations_fail() -> Result<()> {
    let f = Fixture::new()?;
    let path = f.root.join("inventory.json");
    let original: Value = read(&path)?;
    for kind in 0..11 {
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
            _ => v["inputs"].as_array_mut().unwrap().reverse(),
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
    for kind in 0..4 {
        let mut x = input(&f.planned[0])?;
        match kind {
            0 => x.input.observations.reverse(),
            1 => x.input.observations[0][1] = 6,
            2 => x.policies[1].batch += 1,
            _ => x.input.policy.equations -= 1,
        }
        assert!(stage::replay_with(&f.root, &f.planned[0], &inv, x).is_err());
    }
    let a = input("name-a")?;
    let mut changed = task();
    changed.splits.get_mut("validation").unwrap()[0].targets[1] = 6;
    let b = inputs::assemble(changed, "name-b", 2, json!({"synthetic":true}))?;
    assert_eq!(a.input, b.input);
    assert_eq!(a.policies, b.policies);
    for prime in [2, 7] {
        let d = discovery::run(discovery::Request {
            input: a.input.clone(),
            prime,
        })?;
        assert_eq!(
            binding::run(&d, a.policies.clone())?,
            binding::run(&d, b.policies.clone())?
        );
    }
    Ok(())
}
#[test]
fn numerical_mutations_fail_after_rehashing_the_entire_retained_ledger() -> Result<()> {
    let f = Fixture::new()?;
    f.run_first()?;
    let dir = f.root.join(&f.planned[0]);
    let path = dir.join("stage-00-p2.json");
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
        result["stages"][0] = json!({"prime":2,"file":"stage-00-p2.json","sha256":digest(&path)?,
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
            "all_observed_equation_cap"
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
    changed["all_complete_bases_replayed"] = json!(false);
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
    assert_eq!(p["required_done_tasks"], json!([250, 251]));
    assert_eq!(p["core_commit"], "9b10d88091c04cee172e9958c35630c371faedba");
    assert_eq!(p["memory_limit_bytes"], 68719476736_u64);
    assert_eq!(discovery::Policy::default().equations, 8190);
    assert_eq!(discovery::Policy::default().elimination.nullity, 8);
    for name in planned()? {
        assert!(tasks(&suite()?)?.contains(&name));
    }
    assert!(name("../outside").is_err());
    Ok(())
}

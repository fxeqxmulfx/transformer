use super::*;
mod fixtures;
use fixtures::{Fixture, input, overwrite};

#[test]
fn heldout_labels_and_task_names_cannot_change_the_training_request() -> Result<()> {
    let original = input("fixture-original")?;
    let mut changed = fixtures::task();
    for e in changed.splits.get_mut("validation").unwrap() {
        e.tokens.fill(0);
        e.targets[1] = 0;
        e.task = "different-name".into();
    }
    let altered = inputs::assemble(
        changed,
        "fixture-renamed",
        13,
        json!({"different_heldout_fingerprint":true}),
    )?;
    assert_eq!(original.input, altered.input);
    assert_eq!(original.policies, altered.policies);
    assert_eq!(
        original.record["input_sha256"],
        altered.record["input_sha256"]
    );
    Ok(())
}

#[test]
fn production_pins_match_the_unchanged_frozen_parent_source() -> Result<()> {
    let p = protocol()?;
    assert_eq!(p["source_fingerprint"], parent::fingerprint());
    assert_eq!(p["tasks"].as_array().unwrap().len(), 23);
    let reference = Reference {
        root: PathBuf::new(),
        binary: PathBuf::new(),
        inventory_sha256: String::new(),
        source_fingerprint: p["source_fingerprint"].as_str().unwrap().into(),
        binary_sha256: p["source_binary_sha256"].as_str().unwrap().into(),
        producer_commit: p["source_producer"].as_str().unwrap().into(),
    };
    source::pins(&reference)?;
    for kind in 0..3 {
        let mut bad = reference.clone();
        match kind {
            0 => bad.source_fingerprint.push('0'),
            1 => bad.binary_sha256.push('0'),
            _ => bad.producer_commit.push('0'),
        }
        assert!(source::pins(&bad).is_err());
    }
    Ok(())
}

#[test]
fn a_new_process_failure_keeps_partial_work_and_the_complete_task_scope() -> Result<()> {
    let f = Fixture::new()?;
    let inv = f.prepare()?;
    let task = &f.planned[0];
    fs::create_dir(f.destination.join(task))?;
    write(
        &f.destination.join(task).join("partial.json"),
        &json!({"actual_partial_work":123}),
    )?;
    fixtures::process_files(&f.destination, task, false, None)?;
    let failed = summary::case_record(&f.destination, task, &inv, input(task)?)?;
    assert_eq!(failed["status"], "process_failure");
    assert!(
        failed["artifacts"]
            .as_array()
            .unwrap()
            .iter()
            .any(|a| a["path"].as_str().unwrap().ends_with("partial.json"))
    );
    let second = f.execute(&inv, &f.planned[1])?;
    let out = summary::render(&f.destination, &inv, vec![failed, second])?;
    assert_eq!(out["constructor_accounting"]["planned_cases"], 2);
    assert_eq!(out["constructor_accounting"]["process_failures"], 1);
    assert_eq!(
        out["constructor_accounting"]["partial_costs_in_retained_resources"],
        true
    );
    assert_eq!(out["tasks"].as_array().unwrap().len(), 35);
    Ok(())
}
#[test]
fn complete_catalogue_and_failed_source_keep_every_stage_mode_and_cost() -> Result<()> {
    let f = Fixture::new()?;
    let inv = f.prepare()?;
    let first = f.execute(&inv, &f.planned[0])?;
    let failed = f.execute(&inv, &f.planned[1])?;
    assert_eq!(first["result"]["stages"].as_array().unwrap().len(), 55);
    assert_eq!(
        failed["result"]["source_refusal"],
        "original source case failed"
    );
    let summary = summary::render(&f.destination, &inv, vec![first, failed])?;
    assert_eq!(summary["constructor_accounting"]["complete_stages"], 55);
    assert_eq!(summary["constructor_accounting"]["new_discovery_work"], 0);
    assert!(
        summary["constructor_accounting"]["available_endpoints_by_mode"]
            .as_array()
            .unwrap()
            .iter()
            .all(|n| n.as_u64().unwrap() > 0)
    );
    assert_eq!(summary["tasks"].as_array().unwrap().len(), 35);
    assert_eq!(summary["heldout_evaluation"], false);
    assert_eq!(summary["parent_comparison"], false);
    assert!(summary.get("primary_pass").is_none());
    Ok(())
}
#[test]
fn input_order_labels_batch_and_inventory_scope_mutations_are_rejected() -> Result<()> {
    let f = Fixture::new()?;
    let inv = f.prepare()?;
    for kind in 0..4 {
        let mut x = input(&f.planned[0])?;
        match kind {
            0 => {
                x.input.observations.reverse();
                x.record["input_sha256"] = json!(hash(&serde_json::to_vec(&x.input)?));
            }
            1 => {
                x.input.observations[0][1] += 1;
                x.record["input_sha256"] = json!(hash(&serde_json::to_vec(&x.input)?));
            }
            2 => {
                x.policies[1].batch += 1;
                x.record["policies"] = json!(x.policies);
            }
            _ => x.record["task"] = json!("wrong-task"),
        }
        assert!(stage::run_with(&f.destination, &f.planned[0], &inv, x).is_err());
    }
    for key in [
        "phase",
        "source_fingerprint",
        "binary_sha256",
        "tasks",
        "protocol",
        "parent_comparison",
    ] {
        let mut bad = inv.clone();
        bad[key] = json!("changed");
        overwrite(&f.destination.join("inventory.json"), &bad);
        assert!(inventory_with(&f.destination, &f.suite, &f.planned).is_err());
    }
    overwrite(&f.destination.join("inventory.json"), &inv);
    assert!(stage::run_with(&f.destination, "../outside", &inv, input(&f.planned[0])?).is_err());
    Ok(())
}
#[test]
fn source_catalogue_and_replay_evidence_mutations_are_rejected() -> Result<()> {
    let f = Fixture::new()?;
    let inv = f.prepare()?;
    let task = &f.planned[0];
    for relative in [
        format!("{task}.replay.json"),
        format!("{task}/catalogue/configuration.json"),
        format!("{task}/catalogue/ledger.json"),
        format!("{task}/catalogue/stage-00-p2.json"),
    ] {
        let path = f.request.discovery.root.join(relative);
        let bytes = fs::read(&path)?;
        let mut changed = bytes.clone();
        changed.push(b' ');
        fs::write(&path, &changed)?;
        assert!(stage::run_with(&f.destination, task, &inv, input(task)?).is_err());
        fs::write(&path, bytes)?;
    }
    let mut bad = f.request.discovery.clone();
    bad.binary_sha256.push('0');
    assert!(source::validate_reference(&bad, &f.planned, &tasks(&f.suite)?).is_err());
    assert!(source::pins(&f.request.discovery).is_err()); // Fixture executable cannot masquerade as frozen232.
    Ok(())
}
#[test]
fn numerical_mutations_fail_even_after_refined_file_and_ledger_rehash() -> Result<()> {
    let f = Fixture::new()?;
    let inv = f.prepare()?;
    let task = &f.planned[0];
    f.execute(&inv, task)?;
    let dir = f.destination.join(task);
    let original: Value = read(&dir.join("result.json"))?;
    let (index, saved) = original["stages"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .find(|(_, s)| s["retained"]["endpoints"].is_array())
        .unwrap();
    let path = dir.join(saved["file"].as_str().unwrap());
    let initial: stage::Attempt = read(&path)?;
    for kind in 0..4 {
        let mut a = initial.clone();
        let report = a.report.as_mut().unwrap();
        match kind {
            0 => {
                report.endpoints[0]
                    .training
                    .as_mut()
                    .unwrap()
                    .model
                    .coefficients[0] += 0.2
            }
            1 => report.endpoints[1].training.as_mut().unwrap().steps[0]
                .table_sha256
                .push('0'),
            2 => {
                report.endpoints[0]
                    .training
                    .as_mut()
                    .unwrap()
                    .work
                    .gradient_examples += 1
            }
            _ => report.raw_label_identification = true,
        }
        overwrite(&path, &a);
        let mut result = original.clone();
        result["stages"][index]["sha256"] = json!(digest(&path)?);
        result["stages"][index]["bytes"] = json!(fs::metadata(&path)?.len());
        // Rebuild the cheap ledger too, forcing the actual independent replay to reject it.
        result["stages"][index]["retained"] = stage::retained(&a)?;
        overwrite(&dir.join("result.json"), &result);
        assert!(stage::replay_with(&f.destination, task, &inv, input(task)?).is_err());
    }
    overwrite(&path, &initial);
    overwrite(&dir.join("result.json"), &original);
    stage::replay_with(&f.destination, task, &inv, input(task)?)?;
    Ok(())
}

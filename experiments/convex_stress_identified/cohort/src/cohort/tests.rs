use super::*;
use fixtures::*;

#[test]
fn frozen_policy_keeps_all_original_tasks_data_and_memory_limits() -> Result<()> {
    let prior: Value = serde_json::from_str(include_str!(
        "../../../../convex_stress_modular/cohort/protocol.json"
    ))?;
    let p = protocol()?;
    assert_eq!(tasks()?.len(), 23);
    assert_eq!(modes()?.len(), 3);
    for key in [
        "tasks",
        "coding",
        "minimum_strip_support",
        "max_views",
        "shared_fingerprint",
        "memory_limit_bytes",
    ] {
        assert_eq!(p[key], prior[key], "{key}");
    }
    assert_eq!(policy()?, adapter::Policy::default());
    assert_eq!(
        p["shared_fingerprint"],
        convex_stress_relabel::cohort::fingerprint()
    );
    assert_eq!(p["memory_limit_bytes"], 64u64 * 1024 * 1024 * 1024);
    Ok(())
}

#[test]
fn equal_ce_original_models_use_certificates_before_split_evaluation() -> Result<()> {
    let root = temporary();
    fs::create_dir(&root)?;
    let x = fixture("fixture")?;
    let inv = inv(&x);
    let raw = serde_json::to_vec(&search::run(x.input.clone())?)?;
    let parent = parent_fixture(&x, &raw)?;
    run::execute(&root, "fixture", &inv, &parent, x, &raw)?;
    let replay = run::replay_with(&root, "fixture", &inv, &parent, fixture("fixture")?)?;
    assert_eq!(replay["full_modular_parent_replay"], true);
    assert_eq!(replay["categorical_certificate_replay"], true);
    let r: Value = read(&root.join("fixture/result.json"))?;
    assert_eq!(r["certificate_count"], 1);
    assert!(
        r["evaluation"]["choices"]
            .as_array()
            .unwrap()
            .iter()
            .all(|c| c["selected"]["family"] == "identified_modular"
                && c["all_splits_pass"] == true
                && c["baseline"]["all_splits_pass"] == false)
    );
    assert_eq!(bytes(&root.join("fixture/proposal.json"))?, raw);
    assert!(run::execute(&root, "fixture", &inv, &parent, fixture("fixture")?, &raw).is_err());
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn rehashed_certificate_mutations_and_changed_original_inputs_are_rejected() -> Result<()> {
    let root = temporary();
    fs::create_dir(&root)?;
    let x = fixture("fixture")?;
    let inv = inv(&x);
    let raw = serde_json::to_vec(&search::run(x.input.clone())?)?;
    let parent = parent_fixture(&x, &raw)?;
    run::execute(&root, "fixture", &inv, &parent, x, &raw)?;
    let dir = root.join("fixture");
    let mut report: adapter::Report = read(&dir.join("report.json"))?;
    report.entries[0]
        .certificate
        .kernel
        .as_mut()
        .unwrap()
        .work
        .gcd_steps += 1;
    fs::write(dir.join("report.json"), serde_json::to_vec(&report)?)?;
    let mut seal: Value = read(&dir.join("selection.json"))?;
    seal["report_sha256"] = hash(&bytes(&dir.join("report.json"))?).into();
    fs::write(dir.join("selection.json"), serde_json::to_vec(&seal)?)?;
    let mut result: Value = read(&dir.join("result.json"))?;
    result["selection"] = seal;
    result["selection_sha256"] = hash(&bytes(&dir.join("selection.json"))?).into();
    fs::write(dir.join("result.json"), serde_json::to_vec(&result)?)?;
    assert!(run::replay_with(&root, "fixture", &inv, &parent, fixture("fixture")?).is_err());
    let mut changed: search::Report = serde_json::from_slice(&raw)?;
    changed.work.finite_scalar_updates += 1;
    assert!(
        parent::proposal(
            &serde_json::to_vec(&changed)?,
            &parent,
            "fixture",
            &fixture("fixture")?
        )
        .is_err()
    );
    let mut changed = fixture("fixture")?;
    changed.record["data_fingerprints"]["train"] = "changed".into();
    assert!(check_input(&inv, "fixture", &changed).is_err());
    assert!(parent::choices(&parent, "fixture", &changed.record).is_err());
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn all_failed_certifications_retain_every_one_of105_parent_choices() -> Result<()> {
    let root = temporary();
    fs::create_dir(&root)?;
    let parent = full_parent("", json!({}))?;
    for name in tasks()? {
        fs::write(root.join(format!("{name}.run.exit")), "1\n")?;
        fs::write(root.join(format!("{name}.log")), "fixture failure\n")?;
    }
    let summary = summary::summarize_with(
        &root,
        &json!({"producer_commit":"fixture","parent_sha256":"fixture"}),
        parent.clone(),
    )?;
    assert_eq!(summary["choices"], parent["choices"]);
    assert_eq!(summary["process_failures"], 23);
    assert_eq!(summary["constructor_accounting"]["mode_references"], 69);
    assert_eq!(
        summary["constructor_accounting"]["new_model_constructions"],
        0
    );
    assert!(
        summary["by_mode"]
            .as_array()
            .unwrap()
            .iter()
            .all(|r| r["cases"] == 35 && r["no_model"] == 35)
    );
    let mut bad = parent;
    bad["choices"].as_array_mut().unwrap().pop();
    assert!(parent::validate(&bad).is_err());
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn successful_summary_preserves_all105_choices_and_checks_every_artifact() -> Result<()> {
    let root = temporary();
    fs::create_dir(&root)?;
    let name = tasks()?[0].clone();
    let x = fixture(&name)?;
    let inv = inv(&x);
    let raw = serde_json::to_vec(&search::run(x.input.clone())?)?;
    let parent = full_parent(&name, parent_fixture(&x, &raw)?)?;
    run::execute(&root, &name, &inv, &parent, x, &raw)?;
    let seal = run::replay_with(&root, &name, &inv, &parent, fixture(&name)?)?;
    write(&root.join(format!("{name}.replay.json")), &seal)?;
    fs::write(root.join(format!("{name}.replay.exit")), "0\n")?;
    for task in tasks()? {
        fs::write(
            root.join(format!("{task}.run.exit")),
            if task == name { "0\n" } else { "1\n" },
        )?;
        fs::write(root.join(format!("{task}.log")), "fixture outcome\n")?;
    }
    let summary = summary::summarize_with(&root, &inv, parent.clone())?;
    assert_eq!(summary["process_failures"], 22);
    assert_eq!(summary["choices_compared"], 105);
    assert!(
        summary["by_mode"]
            .as_array()
            .unwrap()
            .iter()
            .all(|r| r["cases"] == 35 && r["all_split_passes"] == 1)
    );
    assert_eq!(summary["complete_parent"], parent);
    fs::write(root.join(&name).join("proposal.json"), "{}")?;
    assert!(summary::summarize_with(&root, &inv, parent).is_err());
    let large = root.join("oversized.json");
    fs::File::create(&large)?.set_len(LIMIT + 1)?;
    assert!(bytes(&large).is_err());
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn complete_parent_requires_the_exact_source_and_every_terminal_original_record() -> Result<()> {
    let mut parent = full_parent("", json!({}))?;
    let p = protocol()?;
    parent["source_fingerprint"] = p["parent_source_fingerprint"].clone();
    parent["producer_commit"] = p["parent_producer_commit"].clone();
    parent["tasks"] = 23.into();
    parent["process_failures"] = 0.into();
    parent["records"]=tasks()?.iter().map(|name|json!({"task":name,"status":"complete","seal":{"replayed":true},
        "result":{"metadata":{"source_fingerprint":p["parent_source_fingerprint"],"producer_commit":p["parent_producer_commit"]}}})).collect();
    parent::original(&parent)?;
    for slot in 0..5 {
        let mut bad = parent.clone();
        match slot {
            0 => bad["source_fingerprint"] = "changed".into(),
            1 => bad["producer_commit"] = "changed".into(),
            2 => bad["records"][0]["status"] = "process_failure".into(),
            3 => bad["records"][0]["seal"]["replayed"] = false.into(),
            _ => bad["records"][0] = bad["records"][1].clone(),
        }
        assert!(parent::original(&bad).is_err(), "parent mutation{slot}");
    }
    Ok(())
}

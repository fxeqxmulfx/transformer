use super::*;
use fixtures::*;

#[test]
fn frozen_protocol_keeps_original_tasks_modes_data_memory_and_tested_policy() -> Result<()> {
    let previous: Value = serde_json::from_str(include_str!(
        "../../../convex_stress_identified/cohort/protocol.json"
    ))?;
    let p = protocol()?;
    for key in [
        "tasks",
        "reference_modes",
        "minimum_strip_support",
        "max_views",
        "shared_fingerprint",
        "memory_limit_bytes",
    ] {
        assert_eq!(p[key], previous[key], "{key}");
    }
    assert_eq!(tasks()?.len(), 23);
    assert_eq!(modes()?.len(), 3);
    assert_eq!(policy()?, learner::Policy::default());
    assert_eq!(
        p["shared_fingerprint"],
        convex_stress_relabel::cohort::fingerprint()
    );
    assert_eq!(p["memory_limit_bytes"], 64u64 * 1024 * 1024 * 1024);
    assert_eq!(p["artifact_file_limit_bytes"], LIMIT);
    assert_eq!(fingerprint().len(), 64);
    Ok(())
}
#[test]
fn streaming_views_seal_observed_selection_then_replay_every_model_and_metric() -> Result<()> {
    let root = temporary();
    fs::create_dir(&root)?;
    let x = fixture("fixture")?;
    let inv = inv(&x);
    let parents = parents(&x)?;
    run::execute(&root, "fixture", &inv, parents.clone(), x)?;
    let replay = run::replay_with(&root, "fixture", &inv, &parents, fixture("fixture")?)?;
    assert_eq!(replay["all_view_reports_replayed"], true);
    let dir = root.join("fixture");
    let result: Value = read(&dir.join("result.json"))?;
    let bank: Bank = read(&dir.join("bank.json"))?;
    assert_eq!(bank.views.len(), 2);
    assert!(!bank.candidates.is_empty());
    for r in &bank.views {
        artifacts::verify(&dir, &r.artifact)?;
    }
    assert!(
        result["evaluation"]["choices"]
            .as_array()
            .unwrap()
            .iter()
            .all(|c| c["selected"]["family"] == "decoded_consequences"
                && c["all_splits_pass"] == true
                && c["baseline"]["all_splits_pass"] == false)
    );
    assert!(run::execute(&root, "fixture", &inv, parents, fixture("fixture")?).is_err());
    fs::remove_dir_all(root)?;
    Ok(())
}
#[test]
fn selection_preserves_prior_certificates_and_ignores_heldout_scores() -> Result<()> {
    let x = fixture("fixture")?;
    let request = view::Request {
        input: x.input.clone(),
        view: 1,
    };
    let report = view::run(request)?;
    let mut bank = Bank {
        views: vec![],
        candidates: vec![],
    };
    run::append(
        &mut bank,
        &report,
        artifacts::Artifact {
            file: "view-1.json".into(),
            bytes: 0,
            sha256: "fixture".into(),
        },
    )?;
    let mut parents = parents(&x)?;
    let before = selection::select(&bank, &parents)?;
    for p in &mut parents {
        p["selected"]["all_splits_pass"] = true.into();
    }
    for c in &mut bank.candidates {
        c.value.raw_audit["held_out_accuracy"] = json!(0.);
    }
    assert_eq!(selection::select(&bank, &parents)?, before);
    for p in &mut parents {
        p["sealed_reference"] = json!({"selected":"parent","candidates":[{"id":"parent","model_sha256":"old-exact-fit","certified_within_fixed_view":true}]});
    }
    assert!(
        selection::select(&bank, &parents)?
            .as_array()
            .unwrap()
            .iter()
            .all(|r| r["selected"] == "parent")
    );
    for p in &mut parents {
        p["selected"]["train_ce"] = json!(0.);
    }
    assert!(
        selection::select(&bank, &parents)?
            .as_array()
            .unwrap()
            .iter()
            .all(|r| r["selected"] == "parent")
    );
    Ok(())
}
#[test]
fn rehashed_row_certificate_mutations_and_changed_original_inputs_fail_replay() -> Result<()> {
    let root = temporary();
    fs::create_dir(&root)?;
    let x = fixture("fixture")?;
    let inv = inv(&x);
    let parents = parents(&x)?;
    run::execute(&root, "fixture", &inv, parents.clone(), x)?;
    let dir = root.join("fixture");
    let file = "view-1.json";
    let mut report: view::Report = read(&dir.join(file))?;
    report.interior.as_mut().unwrap().candidates[0]
        .certificate
        .queries[63]
        .label = 0;
    fs::write(dir.join(file), serde_json::to_vec(&report)?)?;
    let mut bank: Bank = read(&dir.join("bank.json"))?;
    bank.views[1].artifact = artifacts::record(&dir, file)?;
    fs::write(dir.join("bank.json"), serde_json::to_vec(&bank)?)?;
    let refs = selection::select(&bank, &parents)?;
    let seal = selection::seal(&dir, &bank, &refs)?;
    fs::write(dir.join("selection.json"), serde_json::to_vec(&seal)?)?;
    let mut result: Value = read(&dir.join("result.json"))?;
    result["selection"] = seal;
    result["views"] = serde_json::to_value(&bank.views)?;
    result["selection_sha256"] = digest(&dir.join("selection.json"))?.into();
    fs::write(dir.join("result.json"), serde_json::to_vec(&result)?)?;
    assert!(run::replay_with(&root, "fixture", &inv, &parents, fixture("fixture")?).is_err());
    let mut x = fixture("fixture")?;
    x.record["data_fingerprints"]["train"] = "changed".into();
    assert!(check_input(&inv, "fixture", &x).is_err());
    fs::remove_dir_all(root)?;
    Ok(())
}
#[test]
fn all_failed_processes_keep_every_previous_choice_and_full_parent() -> Result<()> {
    let root = temporary();
    fs::create_dir(&root)?;
    statuses(&root, None)?;
    let parent = full_parent(None)?;
    let summary = summary::summarize_with(
        &root,
        &json!({"producer_commit":"fixture","parent_sha256":"fixture"}),
        parent.clone(),
    )?;
    assert_eq!(summary["choices"], parent["choices"]);
    assert_eq!(summary["complete_parent"], parent);
    assert_eq!(summary["process_failures"], 23);
    assert_eq!(summary["constructor_accounting"]["mode_references"], 69);
    assert!(
        summary["by_mode"]
            .as_array()
            .unwrap()
            .iter()
            .all(|r| r["cases"] == 35 && r["no_model"] == 35)
    );
    fs::remove_dir_all(root)?;
    Ok(())
}
#[test]
fn complete_summary_preserves105_choices_and_rejects_changed_dense_artifacts() -> Result<()> {
    let root = temporary();
    fs::create_dir(&root)?;
    let name = tasks()?[0].clone();
    let x = fixture(&name)?;
    let inv = inv(&x);
    let parents = parents(&x)?;
    let parent = full_parent(Some(&x))?;
    run::execute(&root, &name, &inv, parents.clone(), x)?;
    let seal = run::replay_with(&root, &name, &inv, &parents, fixture(&name)?)?;
    write(&root.join(format!("{name}.replay.json")), &seal)?;
    statuses(&root, Some(&name))?;
    let out = summary::summarize_with(&root, &inv, parent.clone())?;
    assert_eq!(out["choices_compared"], 105);
    assert_eq!(out["process_failures"], 22);
    assert_eq!(out["complete_parent"], parent);
    assert!(
        out["by_mode"]
            .as_array()
            .unwrap()
            .iter()
            .all(|r| r["cases"] == 35 && r["all_split_passes"] == 1)
    );
    fs::write(root.join(&name).join("view-1.json"), "{}")?;
    assert!(summary::summarize_with(&root, &inv, parent).is_err());
    fs::remove_dir_all(root)?;
    Ok(())
}
#[test]
fn original_parent_requires_exact_freeze_and_all_terminal_records() -> Result<()> {
    let mut parent = full_parent(None)?;
    let p = protocol()?;
    parent["source_fingerprint"] = p["parent_source_fingerprint"].clone();
    parent["producer_commit"] = p["parent_producer_commit"].clone();
    parent["tasks"] = 23.into();
    parent["process_failures"] = 0.into();
    parent["records"]=tasks()?.iter().map(|name| json!({"task":name,"status":"complete","seal":{"replayed":true},
        "result":{"metadata":{"source_fingerprint":p["parent_source_fingerprint"],"producer_commit":p["parent_producer_commit"]}}})).collect();
    parent::original(&parent)?;
    for slot in 0..6 {
        let mut bad = parent.clone();
        match slot {
            0 => bad["source_fingerprint"] = "changed".into(),
            1 => bad["producer_commit"] = "changed".into(),
            2 => bad["records"][0]["status"] = "process_failure".into(),
            3 => bad["records"][0]["seal"]["replayed"] = false.into(),
            4 => bad["records"][0] = bad["records"][1].clone(),
            _ => {
                bad["choices"].as_array_mut().unwrap().pop();
            }
        }
        assert!(parent::original(&bad).is_err(), "mutation{slot}");
    }
    Ok(())
}
#[test]
fn streaming_artifact_hash_matches_reference_and_size_limits_precede_allocation() -> Result<()> {
    let root = temporary();
    fs::create_dir(&root)?;
    let path = root.join("value.json");
    let value = json!({"a":[0,1,2],"b":"original"});
    write(&path, &value)?;
    assert_eq!(digest(&path)?, hash(&fs::read(&path)?));
    assert_eq!(read::<Value>(&path)?, value);
    assert!(write(&path, &value).is_err());
    assert!(artifacts::record(&root, "../value.json").is_err());
    let large = root.join("large.json");
    fs::File::create(&large)?.set_len(LIMIT + 1)?;
    assert!(digest(&large).is_err());
    assert!(read::<Value>(&large).is_err());
    fs::remove_dir_all(root)?;
    Ok(())
}

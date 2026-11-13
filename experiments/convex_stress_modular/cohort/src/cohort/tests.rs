use super::*;
use std::{
    collections::BTreeMap,
    sync::atomic::{AtomicUsize, Ordering},
};
static NEXT: AtomicUsize = AtomicUsize::new(0);
fn temporary() -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "modular-cohort-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ))
}
fn fixture() -> Result<Inputs> {
    let n = 8;
    let mut splits = BTreeMap::new();
    for name in ["train", "validation"] {
        let examples = (0..n * n)
            .filter(|&cell| {
                if name == "train" {
                    cell != 11 && cell != 27
                } else {
                    cell == 11 || cell == 27
                }
            })
            .map(|cell| {
                let (i, j) = (cell % n, cell / n);
                let label = if i == 0 || j == 0 {
                    0
                } else {
                    1 + (i - 1 + j - 1) % 6
                };
                data::Example {
                    task: "fixture".into(),
                    tokens: vec![j as i64, i as i64],
                    targets: vec![-100, label as i64],
                    changes: vec![false, true],
                    prompt: vec![],
                    answer: vec![],
                    limit: 0,
                    field: None,
                    row_index: None,
                }
            })
            .collect();
        splits.insert(name.into(), examples);
    }
    let mut x = assemble(
        data::Task {
            splits,
            vocab: n,
            selection: "validation".into(),
        },
        "fixture",
        json!({"train":"a","validation":"b"}),
    )?;
    x.input.minimum_strip_support = 3;
    x.record["training_input_sha256"] = hash(&serde_json::to_vec(&x.input)?).into();
    Ok(x)
}
fn parent_fixture(x: &Inputs, ce: f64) -> Result<Value> {
    Ok(
        json!({"choices":modes()?.iter().map(|mode|json!({"task":"fixture","training_mode":mode,"selection":"validation","tier":"fixture",
        "status":"selected","primary_pass":false,"all_splits_pass":false,"selected":{"id":"old","train_ce":ce,"model_sha256":"old-model",
            "primary_pass":false,"all_splits_pass":false,"metrics":{"train":{"loss":ce},"validation":{"sequence_accuracy":0.}}}})).collect::<Vec<_>>(),
        "parents":{"complete_general_comparison":{"parents":{"hull_inventory":{"tasks":[{"task":"fixture","vocab":x.task.vocab,"scale":1.0,"data_fingerprints":x.record["data_fingerprints"]}]}}}}}),
    )
}
fn inv(x: &Inputs) -> Value {
    json!({"inputs":[x.record],"producer_commit":"0000000000000000000000000000000000000000","binary_sha256":"fixture","parent_sha256":"fixture-parent"})
}

#[test]
fn unchanged_policy_covers_all_original_pair_tasks_without_discovery() -> Result<()> {
    assert_eq!(tasks()?.len(), 23);
    assert_eq!(modes()?.len(), 3);
    assert_eq!(
        serde_json::from_value::<coding::Policy>(protocol()?["coding"].clone())?,
        coding::Policy::default()
    );
    assert_eq!(
        protocol()?["shared_fingerprint"],
        convex_stress_relabel::cohort::fingerprint()
    );
    assert_eq!(
        protocol()?["memory_limit_bytes"],
        64u64 * 1024 * 1024 * 1024
    );
    Ok(())
}

#[test]
fn masked_composite_strips_replay_with_parent_retention_and_observed_ce_selection() -> Result<()> {
    let root = temporary();
    fs::create_dir(&root)?;
    let x = fixture()?;
    let inv = inv(&x);
    let parent = parent_fixture(&x, 3.)?;
    run::execute(&root, "fixture", &inv, &parent, x)?;
    let replay = run::replay_with(&root, "fixture", &inv, &parent, fixture()?)?;
    assert_eq!(replay["full_integer_strip_state_replay"], true);
    let r: Value = read(&root.join("fixture/result.json"))?;
    assert!(
        r["evaluation"]["choices"]
            .as_array()
            .unwrap()
            .iter()
            .all(|c| c["selected"]["family"] == "modular" && c["all_splits_pass"] == true)
    );
    let report: search::Report = read(&root.join("fixture/report.json"))?;
    let tied = parent_fixture(
        &fixture()?,
        report.candidates[0].audit["train_ce"].as_f64().unwrap(),
    )?;
    let old = parent::choices(&tied, "fixture", &fixture()?.record)?;
    assert!(
        parent::select(&report, &old)?
            .as_array()
            .unwrap()
            .iter()
            .all(|r| r["selected"] == "parent")
    );
    assert!(run::execute(&root, "fixture", &inv, &parent, fixture()?).is_err());
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn rehashed_component_tampering_and_parent_data_changes_are_rejected() -> Result<()> {
    let root = temporary();
    fs::create_dir(&root)?;
    let x = fixture()?;
    let inv = inv(&x);
    let parent = parent_fixture(&x, 3.)?;
    run::execute(&root, "fixture", &inv, &parent, x)?;
    let dir = root.join("fixture");
    let mut report: search::Report = read(&dir.join("report.json"))?;
    report.work.finite_scalar_updates += 1;
    fs::write(dir.join("report.json"), serde_json::to_vec(&report)?)?;
    let mut choice: Value = read(&dir.join("selection.json"))?;
    choice["report_sha256"] = hash(&bytes(&dir.join("report.json"))?).into();
    fs::write(dir.join("selection.json"), serde_json::to_vec(&choice)?)?;
    let mut result: Value = read(&dir.join("result.json"))?;
    result["selection"] = choice;
    result["selection_sha256"] = hash(&bytes(&dir.join("selection.json"))?).into();
    fs::write(dir.join("result.json"), serde_json::to_vec(&result)?)?;
    assert!(run::replay_with(&root, "fixture", &inv, &parent, fixture()?).is_err());
    let mut parent = parent;
    parent["parents"]["complete_general_comparison"]["parents"]["hull_inventory"]["tasks"][0]["scale"] =
        2.into();
    assert!(parent::choices(&parent, "fixture", &fixture()?.record).is_err());
    let mut record = fixture()?.record;
    record["data_fingerprints"]["train"] = "changed".into();
    assert!(
        check_input(
            &inv,
            "fixture",
            &Inputs {
                record,
                ..fixture()?
            }
        )
        .is_err()
    );
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn every_failed_constructor_retains_all105_choices_and_original_mode_counts() -> Result<()> {
    let root = temporary();
    fs::create_dir(&root)?;
    let suite: Value = serde_json::from_str(SUITE)?;
    let mut choices = vec![];
    for task in suite["tasks"].as_array().unwrap() {
        for mode in modes()? {
            choices.push(json!({"task":task["name"],"training_mode":mode,"status":"no_model","selected":null}));
        }
    }
    let parent = json!({"choices_compared":105,"choices":choices,"full35_result":true});
    parent::validate(&parent)?;
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
fn successful_summary_checks_artifacts_and_changes_only_three_of105_choices() -> Result<()> {
    let root = temporary();
    fs::create_dir(&root)?;
    let name = tasks()?[0].clone();
    let mut x = fixture()?;
    x.record["task"] = name.clone().into();
    let inv = inv(&x);
    let mut parent = parent_fixture(&x, 3.)?;
    parent["parents"]["complete_general_comparison"]["parents"]["hull_inventory"]["tasks"][0]["task"] =
        name.clone().into();
    let original = parent["choices"].as_array().unwrap().clone();
    let suite: Value = serde_json::from_str(SUITE)?;
    let mut choices = vec![];
    for task in suite["tasks"].as_array().unwrap() {
        for mode in modes()? {
            if task["name"] == name {
                let mut row = original
                    .iter()
                    .find(|r| r["training_mode"] == mode)
                    .unwrap()
                    .clone();
                row["task"] = name.clone().into();
                choices.push(row);
            } else {
                choices.push(json!({"task":task["name"],"training_mode":mode,"status":"no_model","selected":null}));
            }
        }
    }
    parent["choices"] = choices.into();
    parent["choices_compared"] = 105.into();
    parent["full35_result"] = true.into();
    parent::validate(&parent)?;
    run::execute(&root, &name, &inv, &parent, x)?;
    let mut x = fixture()?;
    x.record["task"] = name.clone().into();
    let seal = run::replay_with(&root, &name, &inv, &parent, x)?;
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
            .all(|m| m["cases"] == 35 && m["all_split_passes"] == 1)
    );
    assert_eq!(summary["complete_parent"], parent);
    fs::write(root.join(&name).join("input.json"), "{}")?;
    assert!(summary::summarize_with(&root, &inv, parent).is_err());
    let large = root.join("oversized.json");
    fs::File::create(&large)?.set_len(LIMIT + 1)?;
    assert!(bytes(&large).is_err());
    fs::remove_dir_all(root)?;
    Ok(())
}

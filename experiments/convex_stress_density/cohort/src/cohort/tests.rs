use super::*;
use std::{
    collections::BTreeMap,
    sync::atomic::{AtomicUsize, Ordering},
};
static NEXT: AtomicUsize = AtomicUsize::new(0);

#[test]
fn summary_keeps_all_planned_mode_references_when_processes_fail() -> Result<()> {
    let root = temporary();
    fs::create_dir(&root)?;
    write(
        &root.join("inventory.json"),
        &json!({"source_fingerprint":fingerprint(),"binary_sha256":binary_fingerprint()?,
        "protocol":protocol()?,"tasks":tasks()?,"inputs":[],
        "producer_commit":"0000000000000000000000000000000000000000"}),
    )?;
    for task in tasks()? {
        fs::write(root.join(format!("{task}.run.exit")), "1\n")?;
        fs::write(
            root.join(format!("{task}.log")),
            "preserved fixture failure\n",
        )?;
    }
    let summary = summarize(&root)?;
    assert_eq!(summary["tasks"], 9);
    assert_eq!(summary["process_failures"], 9);
    assert_eq!(summary["mode_references"].as_array().unwrap().len(), 27);
    assert!(
        summary["mode_references"]
            .as_array()
            .unwrap()
            .iter()
            .all(|r| r["selected"].is_null())
    );
    assert_eq!(summary["full35_result"], false);
    fs::remove_dir_all(root)?;
    Ok(())
}
fn temporary() -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "density-cohort-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ))
}
fn fixture(constant: bool) -> Result<Inputs> {
    let n = 5;
    let truth = Model::construct(
        n,
        vec![3, 1, 4, 0, 2],
        vec![2, 4, 0, 1, 3],
        vec![0, 1, 0, 3, 4],
    )?;
    let mut splits = BTreeMap::new();
    for split in ["train", "validation", "test"] {
        let examples = (0..n * n)
            .map(|cell| data::Example {
                task: "fixture".into(),
                tokens: vec![(cell / n) as i64, (cell % n) as i64],
                targets: vec![
                    -100,
                    if constant {
                        0
                    } else {
                        truth.output_labels[cell] as i64
                    },
                ],
                changes: vec![false, true],
                prompt: vec![],
                answer: vec![],
                limit: 0,
                field: None,
                row_index: None,
            })
            .collect();
        splits.insert(split.into(), examples);
    }
    assemble(
        data::Task {
            splits,
            vocab: n,
            selection: "validation".into(),
        },
        "fixture",
        json!({"fixture":true}),
    )
}
fn inv(x: &Inputs) -> Result<Value> {
    Ok(json!({"inputs":[x.record],"producer_commit":"0000000000000000000000000000000000000000"}))
}

#[test]
fn frozen_policy_matches_measured_fixture_policy_and_bounds() -> Result<()> {
    let p = protocol()?;
    let x = fixture(false)?;
    assert_eq!(tasks()?.len(), 9);
    assert_eq!(x.input.policy, interval_search::Policy::default());
    assert_eq!(
        serde_json::from_value::<refinement_search::Policy>(p["refinement"].clone())?,
        refinement_search::Policy::default()
    );
    assert_eq!(
        p["shared_fingerprint"],
        convex_stress_relabel::cohort::fingerprint()
    );
    assert_eq!(x.input.categorical.options.ranks, [1, 2, 3]);
    assert_eq!(x.input.categorical.options.code_trials, 2);
    assert_eq!(x.input.categorical.options.extraction.iterations, 256);
    assert_eq!(p["memory_limit_bytes"], 68719476736u64);
    Ok(())
}

#[test]
fn end_to_end_fixture_replays_raw_metrics_and_rejects_tampering() -> Result<()> {
    let root = temporary();
    fs::create_dir(&root)?;
    let x = fixture(false)?;
    let inventory = inv(&x)?;
    run::execute(&root, "fixture", &inventory, x)?;
    let seal = run::replay_with(&root, "fixture", &inventory, fixture(false)?)?;
    assert_eq!(seal["raw_latent_ce_metrics_audit"], true);
    let dir = root.join("fixture");
    let r: Value = read(&dir.join("result.json"))?;
    assert_eq!(r["evaluation"]["status"], "complete");
    let chosen = r["evaluation"]["candidates"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["id"] == r["evaluation"]["selected"])
        .unwrap();
    assert_eq!(chosen["all_splits_pass"], true);
    assert!(run::execute(&root, "fixture", &inventory, fixture(false)?).is_err());
    let mut bad = r.clone();
    bad["evaluation"]["selected"] = Value::Null;
    fs::write(dir.join("result.json"), serde_json::to_vec(&bad)?)?;
    assert!(run::replay_with(&root, "fixture", &inventory, fixture(false)?).is_err());
    fs::write(dir.join("result.json"), serde_json::to_vec(&r)?)?;
    let mut report: refinement_search::Report = read(&dir.join("report.json"))?;
    report.assignments_charged += 1;
    fs::write(dir.join("report.json"), serde_json::to_vec(&report)?)?;
    let mut choice: Value = read(&dir.join("selection.json"))?;
    choice["report_sha256"] = hash(&bytes(&dir.join("report.json"))?).into();
    fs::write(dir.join("selection.json"), serde_json::to_vec(&choice)?)?;
    let mut bad = r;
    bad["choice"] = choice;
    bad["selection_sha256"] = hash(&bytes(&dir.join("selection.json"))?).into();
    fs::write(dir.join("result.json"), serde_json::to_vec(&bad)?)?;
    assert!(run::replay_with(&root, "fixture", &inventory, fixture(false)?).is_err());
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn refused_proposals_and_inventory_bounds_remain_explicit() -> Result<()> {
    let root = temporary();
    fs::create_dir(&root)?;
    let x = fixture(true)?;
    let inventory = inv(&x)?;
    run::execute(&root, "fixture", &inventory, x)?;
    let r: Value = read(&root.join("fixture/result.json"))?;
    assert_eq!(r["evaluation"]["status"], "proposal_ineligible");
    assert!(r["evaluation"]["candidates"].as_array().unwrap().is_empty());
    run::replay_with(&root, "fixture", &inventory, fixture(true)?)?;
    assert!(prepare(&root).is_err());
    assert!(inputs("unplanned").is_err());
    let mut altered = inventory.clone();
    altered["inputs"][0]["vocab"] = 6.into();
    assert!(run::replay_with(&root, "fixture", &altered, fixture(true)?).is_err());
    let oversized = root.join("oversized");
    fs::File::create(&oversized)?.set_len(LIMIT + 1)?;
    assert!(bytes(&oversized).is_err());
    fs::remove_dir_all(root)?;
    Ok(())
}

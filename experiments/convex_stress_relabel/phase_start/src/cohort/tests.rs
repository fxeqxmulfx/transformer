use super::*;
use std::{
    collections::BTreeMap,
    time::{Instant, SystemTime, UNIX_EPOCH},
};
const LABELS: [usize; 25] = [
    3, 0, 4, 1, 2, 0, 4, 1, 2, 3, 4, 1, 2, 3, 0, 1, 2, 3, 0, 4, 2, 3, 0, 4, 1,
];
fn fixture(c: &Case) -> Result<Inputs> {
    let example = |cell: usize| cold::data::Example {
        task: c.task.clone(),
        tokens: vec![(cell / 5) as i64, (cell % 5) as i64],
        targets: vec![-100, LABELS[cell] as i64],
        changes: vec![false, true],
        prompt: vec![],
        answer: vec![],
        limit: 0,
        field: None,
        row_index: None,
    };
    let task = cold::data::Task {
        vocab: 5,
        selection: "validation".into(),
        splits: BTreeMap::from([
            (
                "train".into(),
                (0..25).filter(|&x| x != 11).map(example).collect(),
            ),
            ("validation".into(), vec![example(11)]),
            ("test".into(), (0..25).map(example).collect()),
        ]),
    };
    let raw: Vec<_> = (0..25)
        .filter(|&x| x != 11)
        .map(|cell| Sample {
            cell,
            label: LABELS[cell],
        })
        .collect();
    let discovery = cold::coding::discover(5, &raw)?;
    let coding = if c.coding == "identity" {
        Some(cold::Coding::identity(5))
    } else {
        discovery.coding.clone()
    };
    let samples = if let Some(code) = &coding {
        code.samples(&raw)?
    } else {
        vec![]
    };
    Ok(Inputs {
        task,
        row: json!({"data_fingerprints":{"train":"fixture-training","validation":"fixture-withheld"}}),
        discovery,
        coding,
        samples,
        explicit_bytes: cold::memory_bound(5)?,
    })
}
fn temporary() -> Result<std::path::PathBuf> {
    let root = std::env::temp_dir().join(format!(
        "phase-cohort-{}-{}",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    ));
    fs::create_dir(&root)?;
    Ok(root)
}
#[test]
fn end_to_end_fixture_replays_and_rejects_updated_hashes_for_changed_coordinates() -> Result<()> {
    let root = temporary()?;
    let c = cases()?.remove(0);
    let x = fixture(&c)?;
    let inv = json!({"source_fingerprint":fingerprint(),"protocol":protocol()?,"cases":cases()?,
        "producer_commit":"0000000000000000000000000000000000000000","parents":{},"tasks":[task_record(&c.task,&x)]});
    write_new(&root.join("inventory.json"), &inv)?;
    run::execute(&root, c.clone(), inv.clone(), x, Instant::now())?;
    let out = root.join(c.id());
    let result_path = out.join("result.json");
    let model_path = out.join("model.json");
    let original_result = fs::read(&result_path)?;
    let original_model = fs::read(&model_path)?;
    let result: Value = serde_json::from_slice(&original_result)?;
    assert_eq!(result["status"], "complete");
    assert_eq!(result["updates"], 0);
    assert_eq!(result["metrics"]["validation"]["sequence_accuracy"], 1.);
    let seal = replay::replay_with(&root, c.clone(), inv.clone(), fixture(&c)?)?;
    assert_eq!(seal["audit"]["raw_coordinate_audit"], true);
    assert_eq!(seal["audit"]["constructor_reproduced"], true);
    assert_eq!(seal["result_sha256"], hash(&original_result));
    assert!(run::execute(&root, c.clone(), inv.clone(), fixture(&c)?, Instant::now()).is_err());
    let mut model: cold::Model = serde_json::from_slice(&original_model)?;
    let target = model.table.probabilities[55..60]
        .iter()
        .position(|&v| v == 1.)
        .unwrap();
    model.table.probabilities[55..60].fill(0.);
    model.table.probabilities[55 + (target + 1) % 5] = 1.;
    let altered_model = serde_json::to_vec(&model)?;
    fs::write(&model_path, &altered_model)?;
    let mut altered = result.clone();
    altered["model_sha256"] = hash(&altered_model).into();
    fs::write(&result_path, serde_json::to_vec(&altered)?)?;
    assert!(
        replay::replay_with(&root, c.clone(), inv.clone(), fixture(&c)?)
            .unwrap_err()
            .to_string()
            .contains("coordinates")
    );
    fs::write(&model_path, &original_model)?;
    altered = result.clone();
    altered["training_gradient_evaluations"] = 1.into();
    fs::write(&result_path, serde_json::to_vec(&altered)?)?;
    assert!(replay::replay_with(&root, c.clone(), inv.clone(), fixture(&c)?).is_err());
    altered = result.clone();
    altered["moment_norms"][0]["nuclear"] = 0.into();
    fs::write(&result_path, serde_json::to_vec(&altered)?)?;
    assert!(replay::replay_with(&root, c.clone(), inv.clone(), fixture(&c)?).is_err());
    fs::write(&result_path, &original_result)?;
    let mut rejected = c;
    rejected.coding = "identity".into();
    run::execute(
        &root,
        rejected.clone(),
        inv.clone(),
        fixture(&rejected)?,
        Instant::now(),
    )?;
    let rejected_out = root.join(rejected.id());
    assert_eq!(
        read(&rejected_out.join("result.json"))?["status"],
        "proposal_ineligible"
    );
    assert!(!rejected_out.join("model.json").exists());
    let rejected_seal = replay::replay_with(&root, rejected.clone(), inv, fixture(&rejected)?)?;
    assert_eq!(rejected_seal["audit"]["rejected_proposal_reproduced"], true);
    assert!(summarize(&root).is_err());
    fs::remove_dir_all(root)?;
    Ok(())
}
#[test]
fn original_scope_and_incomplete_parent_gate_are_enforced() -> Result<()> {
    let p = protocol()?;
    assert_eq!(
        cold::cohort::fingerprint(),
        p["cold_fingerprint"].as_str().unwrap()
    );
    assert_eq!(
        convex_relabel_graph_audit::fingerprint(),
        p["graph_auditor_fingerprint"].as_str().unwrap()
    );
    let cases = cases()?;
    assert_eq!(cases.len(), 16);
    assert_eq!(
        cases
            .iter()
            .map(Case::id)
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        16
    );
    assert!(
        cases
            .iter()
            .all(|c| c.arm == "phase_construct" && c.budget == "rank_one")
    );
    let root = temporary()?;
    let destination = root.join("fresh");
    assert!(
        prepare(
            &destination,
            &root.join("missing-cold"),
            &root.join("missing-graph")
        )
        .is_err()
    );
    assert!(!destination.exists());
    assert!(replay(&root, "../unplanned").is_err());
    fs::remove_dir_all(root)?;
    Ok(())
}

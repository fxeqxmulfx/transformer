//! Real synthetic runner execution, not a claim about original cohort scores.
use super::*;
use crate::runner::{self, Case, Dataset, Inventory, scoring, stage};
use serde_json::json;
use std::collections::BTreeMap;
const PROCESS: u64 = 16 * 1024 * 1024 * 1024;
fn unavailable(i: usize) -> Case {
    Case {
        name: format!("case-{i}"),
        existing35: false,
        constant_control: false,
        vocabulary: 17,
        training_records: 0,
        validation_records: 0,
        support: 0,
        shape: None,
        audit: None,
        selection_operations: None,
        compact_input_bytes: None,
        input_refusal: None,
        source_refusal: Some("synthetic unavailable control".into()),
        forecast: None,
    }
}
fn refused(c: &Case) -> stage::Stage {
    stage::Stage {
        case: c.name.clone(),
        support: c.support,
        preparation_work: 0,
        preparation_phases: preparation::Phases::new().unwrap(),
        heldout_pools_read: 0,
        endpoints: model::MODES
            .iter()
            .map(|&mode| {
                (
                    model::name(mode).into(),
                    stage::Endpoint {
                        mode,
                        model: None,
                        refusal: c.source_refusal.clone(),
                        charged_work: 0,
                    },
                )
            })
            .collect(),
    }
}
fn write_input<T: serde::Serialize>(f: &Fixture, inv: &mut Inventory, name: &str, value: &T) {
    let path = f.root.join(name);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let a = disk::json(&path, value, runner::JSON_LIMIT, &f.guard).unwrap();
    inv.inputs.insert(name.into(), a);
}
#[test]
fn complete_training_barrier_then_real_scoring_and_replay_keep_both_modes_and_tail() {
    let f = Fixture::new();
    std::fs::create_dir(f.root.join("cases")).unwrap();
    let row = |i: u32| {
        let (a, b) = (i / 17, i % 17);
        [a, b, ((a + b) % 17) % 7]
    };
    let t = pairs::Training {
        vocabulary: 17,
        rows: (0..130).map(row).collect(),
    };
    let data = Dataset {
        training: t.clone(),
        validation: (130..289).map(row).collect(),
    };
    let mut w = Work::new(pairs::MAX_WORK).unwrap();
    let selection = pairs::selection::select(&t, &mut w).unwrap();
    let case = runner::inputs::specification(
        &json!({"case":"synthetic","existing35":false,
        "source_case":{"data":{"prime":17}},"refusal":null,
        "data":{"train":{"rows":130},"validation":{"rows":159}}}),
        Some(&selection),
    )
    .unwrap();
    let mut cases = vec![case];
    cases.extend((1..54).map(unavailable));
    let mut inv = Inventory {
        protocol: runner::protocol().unwrap(),
        producer_commit: "synthetic".into(),
        cases,
        inputs: BTreeMap::new(),
        single_run_budget: f.guard.output_budget,
    };
    write_input(&f, &mut inv, "inputs/train/synthetic.json", &t);
    write_input(&f, &mut inv, "inputs/data/synthetic.json", &data);
    write_input(
        &f,
        &mut inv,
        "inputs/probe/synthetic.json",
        &json!({"selection":selection,"work":w,
        "protocol_sha256":pairs::digest(pairs::PROTOCOL.as_bytes()),"expanded_training_records_allocated":0,
        "heldout_pools_read":0,"accuracy_measurements":0,"training_updates":0}),
    );
    // Synthetic history exercises report field scope, not historical accuracy.
    let history = json!({"cases":inv.cases.iter().map(|c|
        json!({"case":c.name,"historical_gpt":null})).collect::<Vec<_>>()});
    write_input(&f, &mut inv, "inputs/data-inventory.json", &history);
    write_input(&f, &mut inv, "inputs/original35-metrics.json", &json!([]));
    write_input(
        &f,
        &mut inv,
        "inputs/catalog77-historical-verification.json",
        &json!({"catalog_all_modes_all_heldout99_cases":77,"catalog_total_cases":165}),
    );
    disk::json(
        &f.root.join("inventory.json"),
        &inv,
        runner::JSON_LIMIT,
        &f.guard,
    )
    .unwrap();
    let data_path = f.root.join("inputs/data/synthetic.json");
    let original = std::fs::read(&data_path).unwrap();
    // Same length, but impossible to deserialize. Training must not read it.
    std::fs::write(&data_path, vec![b'x'; original.len()]).unwrap();
    stage::execute(&f.root, "synthetic", inv.clone(), f.guard.clone(), PROCESS).unwrap();
    let staged: stage::Stage = runner::read(&f.root.join("cases/synthetic/training.json")).unwrap();
    assert_eq!(staged.endpoints.len(), 2);
    assert!(
        staged.endpoints["forward"]
            .model
            .as_ref()
            .unwrap()
            .steps
            .is_empty()
    );
    let mini = staged.endpoints["mini"].model.as_ref().unwrap();
    assert_eq!(mini.steps.len(), 2);
    assert_eq!(
        (
            mini.steps[1].first_record,
            mini.steps[1].end_record_exclusive
        ),
        (127, 130)
    );
    assert_eq!(staged.preparation_phases.selection.used, w.used);
    assert!(staged.preparation_phases.materialization.used > 0);
    assert!(
        scoring::execute(
            &f.root,
            "synthetic",
            false,
            inv.clone(),
            f.guard.clone(),
            PROCESS
        )
        .is_err()
    );
    assert!(!f.root.join("cases/synthetic/score.json").exists());
    assert!(stage::create_seal(&f.root, &inv, &f.guard).is_err());
    for (i, c) in inv.cases.iter().enumerate().skip(1) {
        let dir = f.root.join("cases").join(&c.name);
        std::fs::create_dir(&dir).unwrap();
        disk::json(
            &dir.join("training.json"),
            &refused(c),
            runner::JSON_LIMIT,
            &f.guard,
        )
        .unwrap();
        if i == 52 {
            assert!(stage::create_seal(&f.root, &inv, &f.guard).is_err());
        }
    }
    std::fs::write(&data_path, original).unwrap();
    stage::create_seal(&f.root, &inv, &f.guard).unwrap();
    let seal: stage::Seal = runner::read(&f.root.join("training-seal.json")).unwrap();
    assert_eq!(
        (seal.cases, seal.endpoints, seal.heldout_pools_scored),
        (54, 108, 0)
    );
    scoring::execute(
        &f.root,
        "synthetic",
        false,
        inv.clone(),
        f.guard.clone(),
        PROCESS,
    )
    .unwrap();
    scoring::execute(
        &f.root,
        "synthetic",
        true,
        inv.clone(),
        f.guard.clone(),
        PROCESS,
    )
    .unwrap();
    let score: scoring::Score = runner::read(&f.root.join("cases/synthetic/score.json")).unwrap();
    assert_eq!(score.preparation_phases, staged.preparation_phases);
    for e in score.endpoints.values() {
        assert_eq!(e.splits["train"].metrics.targets, 130);
        assert_eq!(e.splits["validation"].metrics.targets, 159);
        e.splits["validation"].metrics.validate(159).unwrap();
    }
    for c in inv.cases.iter().skip(1) {
        for replay in [false, true] {
            scoring::execute(
                &f.root,
                &c.name,
                replay,
                inv.clone(),
                f.guard.clone(),
                PROCESS,
            )
            .unwrap();
        }
    }
    let report = runner::summary::build(&f.root, &inv).unwrap();
    assert_eq!(report["required_modes"], json!(["forward", "mini"]));
    assert_eq!(report["optional_reference_modes"], json!(["full"]));
    assert_eq!(report["historical_three_mode_verified_catalog_cases"], 77);
    assert_eq!(report["cases"].as_array().unwrap().len(), 54);
    assert_eq!(report["modes"].as_array().unwrap().len(), 2);
    assert!(report.get("candidate_catalog_union_cases").is_none());
    assert!(report.get("new_all_modes_all_heldout99_cases").is_none());
    let mut changed = staged.clone();
    changed.endpoints.remove("mini");
    assert!(stage::validate_stage(&changed, &inv.cases[0]).is_err());
    changed = staged.clone();
    changed.preparation_phases.materialization.used = 0;
    changed.preparation_work = changed.preparation_phases.total().unwrap();
    assert!(stage::validate_stage(&changed, &inv.cases[0]).is_err());
    let mut changed = score.clone();
    changed.preparation_phases.selection.used -= 1;
    changed.preparation_work = changed.preparation_phases.total().unwrap();
    assert!(
        scoring::validate_score(&changed, &staged, &inv.cases[0], &score.training_seal).is_err()
    );
    std::fs::write(
        f.root.join("cases/synthetic/training.json"),
        serde_json::to_vec(&refused(&inv.cases[1])).unwrap(),
    )
    .unwrap();
    assert!(stage::require_seal(&f.root, &inv).is_err());
}

#[test]
fn production_pins_and_capacity_checks_do_not_treat_input_forecast_as_a_fitted_model() {
    let mut p = runner::protocol().unwrap();
    assert!(runner::validate_production(&p).is_err());
    p["production_ready"] = json!(true);
    p["references"]["probe-manifest.sha256"] = json!("0".repeat(64));
    p["references"]["probe-verified.json"] = json!("1".repeat(64));
    runner::validate_production(&p).unwrap();
    let mut wrong = p.clone();
    wrong["training"]["mini_batch_size"] = json!(128);
    assert!(runner::validate_production(&wrong).is_err());
    let mut wrong = p.clone();
    wrong["metadata_parser_reserve_bytes"] = json!(0);
    assert!(runner::validate_production(&wrong).is_err());
    p["required_modes"] = json!(["forward", "full", "mini"]);
    assert!(runner::validate_production(&p).is_err());
    let shape = Shape {
        programs: 962806,
        states: 196,
        classes: 193,
    };
    let bytes = runner::memory::input_forecast(shape, 18528, 7324366).unwrap();
    assert!(bytes > 35_000_000_000 && bytes < 50_000_000_000);
    assert!(runner::memory::input_forecast(shape, 65536, shape.rows()).is_err());
    let f = Fixture::new();
    let mut inv = Inventory {
        protocol: json!({}),
        producer_commit: "test".into(),
        cases: vec![unavailable(0)],
        inputs: BTreeMap::new(),
        single_run_budget: 1,
    };
    let before = runner::memory::caller(&inv, &f.guard).unwrap();
    inv.cases[0].name.reserve_exact(1024 * 1024);
    assert!(runner::memory::caller(&inv, &f.guard).unwrap() > before + 1024 * 1024 - 100);
    let mut staged = refused(&inv.cases[0]);
    staged
        .endpoints
        .get_mut("mini")
        .unwrap()
        .refusal
        .as_mut()
        .unwrap()
        .reserve_exact(runner::memory::ENDPOINT_METADATA as usize);
    assert!(runner::memory::stage(&staged).is_err());
}

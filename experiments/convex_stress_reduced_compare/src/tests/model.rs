use super::*;
use crate::{evaluation, model, runner};
use joint::{Work, training::Mode};
use serde_json::{Value, json};
use std::collections::BTreeMap;

fn training() -> pairs::Training {
    let labels = [0, 2, 4, 2, 0];
    pairs::Training {
        vocabulary: 5,
        rows: (0..5)
            .flat_map(|a| (0..5).map(move |b| [a, b, labels[((a + b) % 5) as usize]]))
            .collect(),
    }
}
fn prepared(input: &pairs::Training) -> model::Prepared {
    let mut work = Work::new(crate::MAX_WORK).unwrap();
    let selection = pairs::selection::select(input, &mut work).unwrap();
    model::prepare(input, &selection, &mut work).unwrap()
}
#[test]
fn complete_saved_models_keep_actual_full_mini_and_forward_policies() {
    let fixture = Fixture::new();
    let mut input = training();
    input.rows = (0..130).map(|i| input.rows[i % 25]).collect();
    let p = prepared(&input);
    for mode in model::MODES {
        let mut work = Work::new(crate::MAX_WORK).unwrap();
        let trained =
            joint::training::train(&p.domain, &p.records, pairs::policy(mode), &mut work).unwrap();
        let path = fixture.root.join(format!("{}.bin", model::name(mode)));
        let artifact = tensor::save(
            &path,
            &p.domain,
            &trained.parameters,
            &trained.head,
            &fixture.guard,
            &mut work,
        )
        .unwrap();
        let state = model::State::from_trained(
            &trained,
            &p.domain,
            artifact.clone(),
            work.used,
            p.domain
                .check(&trained.parameters, 1e-8, &mut work)
                .unwrap(),
        );
        let (q, head) = tensor::load(
            &path,
            &p.domain,
            &artifact.artifact,
            artifact.body_sha256,
            &mut work,
        )
        .unwrap();
        state.check(&p.domain, &q, &head, &mut work).unwrap();
        match mode {
            Mode::Forward => {
                assert!(state.steps.is_empty());
                assert_eq!(state.initial_ce, state.final_ce);
            }
            Mode::Full => {
                assert_eq!(state.steps.len(), 4);
                assert!(state.final_ce < state.initial_ce);
            }
            Mode::Mini => {
                assert_eq!(state.steps.len(), 2);
                assert_eq!(state.steps[1].first_record, 127);
                assert_eq!(state.steps[1].end_record_exclusive, 130);
                assert!(state.final_ce < state.initial_ce);
            }
        }
        let mut changed = state.clone();
        changed.config.learning_rate = 10.;
        assert!(changed.check(&p.domain, &q, &head, &mut work).is_err());
        let mut changed = state.clone();
        changed.target_realizability_certified = true;
        assert!(changed.check(&p.domain, &q, &head, &mut work).is_err());
    }
}
#[test]
fn raw_counts_keep_wrong_labels_and_99_boundary_and_reject_rehashed_tampering() {
    let fixture = Fixture::new();
    let input = training();
    let p = prepared(&input);
    let mut work = Work::new(crate::MAX_WORK).unwrap();
    let trained = joint::training::train(
        &p.domain,
        &p.records,
        pairs::policy(Mode::Forward),
        &mut work,
    )
    .unwrap();
    let mut rows = vec![input.rows[1]; 100];
    rows[99][2] = (rows[99][2] + 1) % 5;
    let path = fixture.root.join("raw.jsonl");
    let mut predictor =
        model::Predictor::new(&trained.head, &p.domain, &p.selection, 100, &mut work).unwrap();
    let (artifact, metrics) =
        evaluation::score(&path, &rows, 5, &mut predictor, &mut work, &fixture.guard).unwrap();
    assert_eq!(metrics.targets, 100);
    assert_eq!(metrics.raw_correct, 99);
    assert!(metrics.raw99 && metrics.exact99 && metrics.original_success);
    metrics.validate(100).unwrap();
    let mut forged = metrics.clone();
    forged.targets = 99;
    assert!(forged.validate(100).is_err());
    let mut forged = metrics.clone();
    forged.raw_correct = 98;
    assert!(forged.validate(100).is_err());
    let mut forged = metrics.clone();
    forged.ordinary_cross_entropy = None;
    assert!(forged.validate(100).is_err());
    assert!(metrics.ordinary_cross_entropy.unwrap() > 0.08);
    let mut predictor =
        model::Predictor::new(&trained.head, &p.domain, &p.selection, 100, &mut work).unwrap();
    assert_eq!(
        evaluation::replay(&path, &artifact, &rows, 5, &mut predictor, &mut work).unwrap(),
        metrics
    );
    let raw = std::fs::read_to_string(&path).unwrap();
    let mut values: Vec<Value> = raw
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(values[0]["answer"], values[99]["answer"]);
    values[5]["answer"]["numerical"]["logits"][0] = json!(123.);
    let mut output = std::fs::File::create(&path).unwrap();
    for value in values {
        serde_json::to_writer(&mut output, &value).unwrap();
        writeln!(output).unwrap();
    }
    output.sync_all().unwrap();
    let mut predictor =
        model::Predictor::new(&trained.head, &p.domain, &p.selection, 100, &mut work).unwrap();
    assert!(
        evaluation::replay(
            &path,
            &disk::hash(&path).unwrap(),
            &rows,
            5,
            &mut predictor,
            &mut work
        )
        .is_err()
    );
    rows[98][2] = (rows[98][2] + 1) % 5;
    let mut predictor =
        model::Predictor::new(&trained.head, &p.domain, &p.selection, 100, &mut work).unwrap();
    let (_, metrics) = evaluation::score(
        &fixture.root.join("98.jsonl"),
        &rows,
        5,
        &mut predictor,
        &mut work,
        &fixture.guard,
    )
    .unwrap();
    assert_eq!(metrics.raw_correct, 98);
    assert!(!metrics.original_success);
}
#[test]
fn numerical_refusals_keep_all_targets_and_disk_failure_stops_before_scoring() {
    let fixture = Fixture::new();
    let input = training();
    let p = prepared(&input);
    let mut work = Work::new(crate::MAX_WORK).unwrap();
    let trained = joint::training::train(
        &p.domain,
        &p.records,
        pairs::policy(Mode::Forward),
        &mut work,
    )
    .unwrap();
    let mut predictor =
        model::Predictor::new(&trained.head, &p.domain, &p.selection, 25, &mut work).unwrap();
    let mut exhausted = Work::new(1).unwrap();
    let (_, metrics) = evaluation::score(
        &fixture.root.join("refused.jsonl"),
        &input.rows,
        5,
        &mut predictor,
        &mut exhausted,
        &fixture.guard,
    )
    .unwrap();
    assert_eq!(metrics.targets, 25);
    assert_eq!(metrics.refusals, 25);
    assert_eq!(metrics.raw_correct, 0);
    assert!(metrics.ordinary_cross_entropy.is_none());
    assert!(!metrics.original_success);
    let mut denied = fixture.guard.clone();
    denied.storage_reserve = u64::MAX;
    let path = fixture.root.join("disk-stopped.jsonl");
    let mut predictor =
        model::Predictor::new(&trained.head, &p.domain, &p.selection, 25, &mut work).unwrap();
    let mut untouched = Work::new(crate::MAX_WORK).unwrap();
    assert!(
        evaluation::score(
            &path,
            &input.rows,
            5,
            &mut predictor,
            &mut untouched,
            &denied
        )
        .is_err()
    );
    assert_eq!(untouched.used, 0);
    assert!(!path.exists());
}
#[test]
fn training_barrier_rejects_missing_cases_and_changed_sealed_metadata() {
    let fixture = Fixture::new();
    std::fs::create_dir(fixture.root.join("cases")).unwrap();
    let cases: Vec<_> = ["first", "second"]
        .iter()
        .map(|&name| runner::Case {
            name: name.into(),
            existing35: false,
            constant_control: false,
            vocabulary: 5,
            training_records: 0,
            validation_records: 0,
            support: 0,
            shape: None,
            audit: None,
            source_refusal: Some("source unavailable".into()),
            forecast: None,
        })
        .collect();
    let inv = runner::Inventory {
        protocol: json!({}),
        producer_commit: "test".into(),
        cases,
        inputs: BTreeMap::new(),
        single_run_budget: 1024 * 1024,
    };
    disk::json(
        &fixture.root.join("inventory.json"),
        &inv,
        65536,
        &fixture.guard,
    )
    .unwrap();
    assert!(runner::stage::require_seal(&fixture.root, &inv).is_err());
    for (index, case) in inv.cases.iter().enumerate() {
        let dir = fixture.root.join("cases").join(&case.name);
        std::fs::create_dir(&dir).unwrap();
        let endpoints = model::MODES
            .iter()
            .map(|&mode| {
                (
                    model::name(mode).into(),
                    runner::stage::Endpoint {
                        mode,
                        model: None,
                        refusal: Some("source unavailable".into()),
                        charged_work: 0,
                    },
                )
            })
            .collect();
        disk::json(
            &dir.join("training.json"),
            &runner::stage::Stage {
                case: case.name.clone(),
                support: 0,
                preparation_work: 0,
                heldout_pools_read: 0,
                endpoints,
            },
            65536,
            &fixture.guard,
        )
        .unwrap();
        if index == 0 {
            assert!(runner::stage::create_seal(&fixture.root, &inv, &fixture.guard).is_err());
            assert!(!fixture.root.join("training-seal.json").exists());
        }
    }
    runner::stage::create_seal(&fixture.root, &inv, &fixture.guard).unwrap();
    let seal = runner::stage::require_seal(&fixture.root, &inv).unwrap();
    let path = fixture.root.join("cases/first/training.json");
    let mut changed: runner::stage::Stage = runner::read(&path).unwrap();
    let mut score = runner::scoring::Score {
        case: "first".into(),
        training_seal: seal.clone(),
        endpoints: model::MODES
            .iter()
            .map(|&mode| {
                (
                    model::name(mode).into(),
                    runner::scoring::Endpoint {
                        mode,
                        refusal: Some("source unavailable".into()),
                        splits: BTreeMap::new(),
                        charged_work: 0,
                    },
                )
            })
            .collect(),
    };
    runner::scoring::validate_score(&score, &changed, &inv.cases[0], &seal).unwrap();
    score.endpoints.get_mut("forward").unwrap().refusal = None;
    assert!(runner::scoring::validate_score(&score, &changed, &inv.cases[0], &seal).is_err());
    changed.endpoints.get_mut("forward").unwrap().refusal = Some("different refusal".into());
    std::fs::write(path, serde_json::to_vec(&changed).unwrap()).unwrap();
    assert!(runner::stage::require_seal(&fixture.root, &inv).is_err());
}
#[test]
fn original_split_and_protocol_controls_are_explicit() {
    let p = runner::protocol().unwrap();
    assert_eq!(
        p["family_sha256"],
        pairs::digest(&serde_json::to_vec(&pairs::family::programs()).unwrap())
    );
    assert_eq!(p["process_ram_bytes"], RAM_LIMIT_BYTES);
    assert_eq!(
        p["training"]["mini_batch_size"],
        pairs::policy(Mode::Mini).batch_size
    );
    for bad in ["", "/absolute", "../outside", "good/../outside"] {
        assert!(runner::relative(bad).is_err());
    }
    runner::relative("inputs/data/case.json").unwrap();
    let input = training();
    let case = runner::Case {
        name: "fixture".into(),
        existing35: false,
        constant_control: false,
        vocabulary: 5,
        training_records: input.rows.len(),
        validation_records: 1,
        support: 1,
        shape: None,
        audit: None,
        source_refusal: None,
        forecast: None,
    };
    let repeated = runner::Dataset {
        training: input.clone(),
        validation: vec![input.rows[0]],
    };
    assert!(repeated.validate(&input, &case).is_err());
}

#[test]
fn production_requires_complete_prospective_pins() {
    let mut p = runner::protocol().unwrap();
    p["production_ready"] = json!(false);
    assert!(runner::validate_production(&p).is_err());
    p["production_ready"] = json!(true);
    p["references"]["probe-manifest.sha256"] = json!("0".repeat(64));
    p["references"]["probe-verified.json"] = json!("1".repeat(64));
    runner::validate_production(&p).unwrap();
    p["references"]["probe-verified.json"] = json!(null);
    assert!(runner::validate_production(&p).is_err());
    p["references"]["probe-verified.json"] = json!("unverified");
    assert!(runner::validate_production(&p).is_err());
    assert_eq!(
        runner::family_digest(),
        p["family_sha256"].as_str().unwrap()
    );
}

#[test]
fn oversized_complete_support_is_retained_as_an_explicit_model_refusal() {
    let shape = Shape {
        programs: pairs::PROGRAMS,
        states: 196,
        classes: 193,
    };
    let unknown = 3 * shape.programs;
    let columns = shape.programs * (48 * 4 + 193);
    let refusal = shape
        .admit(18528, unknown, columns)
        .unwrap_err()
        .to_string();
    let mut selected = pairs::selection::Selection {
        family: pairs::FAMILY.into(),
        family_sha256: runner::family_digest().into(),
        vocabulary: 193,
        records: 18528,
        original_ordered_training_sha256: "0".repeat(64),
        total_candidates: pairs::PROGRAMS,
        retained: (0..pairs::PROGRAMS).collect(),
        conflicts: vec![],
        audit: pairs::selection::Audit {
            known_rows: shape.rows() - unknown,
            unknown_rows: unknown,
            head_columns_bound: columns,
            dictionary_sha256: "1".repeat(64),
            program_major_states_sha256: "2".repeat(64),
            independently_checked_states: shape.programs as u64 * 18528,
            independently_checked_conflicts: 0,
            complete_buffer_forecast: None,
            admission_refusal: Some(refusal.clone()),
        },
        operations: 1,
    };
    let row = json!({"case":"synthetic-size-control", "existing35":false,
        "source_case":{"data":{"prime":193}},"refusal":null,
        "data":{"train":{"rows":18528},"validation":{"rows":18528}}});
    let case = runner::inputs::specification(&row, Some(&selected)).unwrap();
    assert_eq!(case.support, pairs::PROGRAMS);
    assert_eq!(case.shape, Some(shape));
    assert_eq!(case.validation_records, 18528);
    assert!(case.source_refusal.is_none() && case.forecast.is_none());
    assert_eq!(case.audit.unwrap().admission_refusal, Some(refusal));
    selected.audit.admission_refusal = None;
    assert!(runner::inputs::specification(&row, Some(&selected)).is_err());
    selected.audit.complete_buffer_forecast = Some(1);
    assert!(runner::inputs::specification(&row, Some(&selected)).is_err());
}

#[test]
fn free_unseen_completions_keep_observed_ce_and_report_genuine_ties() {
    let fixture = Fixture::new();
    let input = pairs::Training {
        vocabulary: 5,
        rows: (0..4)
            .flat_map(|a| (0..5).map(move |b| [a, b, a]))
            .collect(),
    };
    let prepared = prepared(&input);
    let mut work = Work::new(crate::MAX_WORK).unwrap();
    let evaluator = pairs::family::Evaluator::new(5).unwrap();
    let bank = pairs::family::programs();
    let index = prepared
        .selection
        .retained
        .iter()
        .position(|&id| (0..5).all(|a| (0..5).all(|b| evaluator.evaluate(bank[id], a, b) == a)))
        .unwrap();
    let domain = &prepared.domain;
    let mut parameters = Parameters {
        values: vec![0.; domain.parameters()],
    };
    parameters.values[index] = 8.;
    for &(p, s) in &domain.unknown {
        if p == index {
            let at = domain.offsets[p * domain.shape.states + s];
            parameters.values[at..at + 5].fill(8. / 5.);
        }
    }
    domain.check(&parameters, 1e-8, &mut work).unwrap();
    let head = Head::build(domain, &parameters, &mut work).unwrap();
    let mut predictor =
        model::Predictor::new(&head, domain, &prepared.selection, 2, &mut work).unwrap();
    let rows = [[4, 0, 0], [4, 0, 4]];
    let (_, metrics) = evaluation::score(
        &fixture.root.join("ambiguous.jsonl"),
        &rows,
        5,
        &mut predictor,
        &mut work,
        &fixture.guard,
    )
    .unwrap();
    assert_eq!(metrics.raw_correct, 1);
    assert_eq!(metrics.genuine_ties, 2);
    assert_eq!(metrics.exact_queries, 2);
    assert!((metrics.ordinary_cross_entropy.unwrap() - 5f64.ln()).abs() < 1e-12);
    let at = domain.offsets[index * domain.shape.states + 4];
    parameters.values[at..at + 5].copy_from_slice(&[0., 0., 0., 0., 8.]);
    domain.check(&parameters, 1e-8, &mut work).unwrap();
    let completed = Head::build(domain, &parameters, &mut work).unwrap();
    assert_eq!(
        joint::training::loss(&head, &prepared.records, &mut work).unwrap(),
        joint::training::loss(&completed, &prepared.records, &mut work).unwrap()
    );
    let mut predictor =
        model::Predictor::new(&completed, domain, &prepared.selection, 1, &mut work).unwrap();
    let answer = predictor.predict([4, 0], &mut work).unwrap();
    assert_eq!(answer.numerical.winner, 4);
    assert_eq!(answer.numerical.maximizers, vec![4]);
    assert!(!answer.numerical.target_realizability_certified);
}

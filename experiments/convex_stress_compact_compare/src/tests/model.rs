use super::*;
use joint::training::Mode;
fn training() -> pairs::Training {
    let labels = [0, 2, 4, 2, 0];
    pairs::Training {
        vocabulary: 5,
        rows: (0..5)
            .flat_map(|a| (0..5).map(move |b| [a, b, labels[((a + b) % 5) as usize]]))
            .collect(),
    }
}
fn prepared(t: &pairs::Training) -> preparation::Prepared {
    let selected = pairs::selection::select(t, &mut Work::new(pairs::MAX_WORK).unwrap()).unwrap();
    preparation::prepare(
        t,
        &selected,
        4 * 1024 * 1024 * 1024,
        16 * 1024 * 1024,
        &mut preparation::Phases::new().unwrap(),
    )
    .unwrap()
}
#[test]
fn required_endpoints_save_load_and_predict_identically_to_the_complete_prior_model() {
    let f = Fixture::new();
    let mut t = training();
    t.rows = (0..130).map(|i| t.rows[i % 25]).collect();
    let p = prepared(&t);
    let mut old_work = Work::new(pairs::MAX_WORK).unwrap();
    let old = reference::model::prepare(&t, &p.selection, &mut old_work).unwrap();
    assert_eq!(p.domain, old.domain);
    for mode in model::MODES {
        let mut w = work::endpoint(0).unwrap();
        let trained = p
            .budget
            .train(&p.domain, &p.records, pairs::policy(mode), &mut w)
            .unwrap();
        let old_trained = legacy::training::train_source(
            &old.domain,
            old.source(),
            pairs::policy(mode),
            &mut old_work,
        )
        .unwrap();
        assert_eq!(trained.steps, old_trained.steps);
        assert_eq!(
            trained.initial_ce.to_bits(),
            old_trained.initial_ce.to_bits()
        );
        assert_eq!(trained.final_ce.to_bits(), old_trained.final_ce.to_bits());
        let residual = trained.parameters.check(&p.domain, 1e-8, &mut w).unwrap();
        let path = f.root.join(format!("{}.bin", model::name(mode)));
        let saved = tensor::save(
            &path,
            &p.domain,
            &trained.parameters,
            &trained.head,
            &f.guard,
            ROOM,
            &mut w,
        )
        .unwrap();
        let state = model::State::from_trained(trained, &p.domain, saved.clone(), w.used, residual);
        let (q, head) = tensor::load(
            &path,
            &p.domain,
            &saved.artifact,
            saved.body_sha256,
            ROOM,
            &mut w,
        )
        .unwrap();
        state.check(&p.domain, &q, &head, ROOM, &mut w).unwrap();
        assert_eq!(
            q.values().map(|v| v.to_bits()).collect::<Vec<_>>(),
            old_trained
                .parameters
                .values
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        if mode == Mode::Mini {
            assert_eq!(state.steps.len(), 2);
            assert_eq!(state.steps[1].first_record, 127);
            assert_eq!(state.steps[1].end_record_exclusive, 130);
            assert!(state.final_ce < state.initial_ce);
        } else {
            assert!(state.steps.is_empty());
            assert_eq!(state.initial_ce, state.final_ce);
        }
        let mut bad = state.clone();
        bad.total_operations = work::ENDPOINT_LIMIT + 1;
        assert!(bad.check(&p.domain, &q, &head, ROOM, &mut w).is_err());
        let mut bad = state.clone();
        bad.target_realizability_certified = true;
        assert!(bad.check(&p.domain, &q, &head, ROOM, &mut w).is_err());
        let before = w.used;
        assert!(model::Predictor::new(&head, &p.domain, &p.selection, 25, 1, &mut w).is_err());
        assert_eq!(before, w.used);
        let mut new =
            model::Predictor::new(&head, &p.domain, &p.selection, 25, ROOM, &mut w).unwrap();
        let mut prior = reference::model::Predictor::new(
            &old_trained.head,
            &old.domain,
            &old.selection,
            25,
            &mut old_work,
        )
        .unwrap();
        for a in 0..5 {
            for b in 0..5 {
                let (start, old_start) = (w.used, old_work.used);
                let x = new.predict([a, b], &mut w).unwrap();
                let y = prior.predict([a, b], &mut old_work).unwrap();
                assert_eq!(
                    serde_json::to_vec(&x).unwrap(),
                    serde_json::to_vec(&y).unwrap()
                );
                assert_eq!(w.used - start, old_work.used - old_start);
            }
        }
    }
}
#[test]
fn raw_replay_preserves_wrong_labels_denominators_refusals_and_rejects_rehashed_corruption() {
    let f = Fixture::new();
    let t = training();
    let p = prepared(&t);
    let mut w = work::endpoint(0).unwrap();
    let trained = p
        .budget
        .train(&p.domain, &p.records, pairs::policy(Mode::Forward), &mut w)
        .unwrap();
    let mut rows = vec![t.rows[1]; 100];
    rows[99][2] = (rows[99][2] + 1) % 5;
    let make = |w: &mut Work| {
        model::Predictor::new(&trained.head, &p.domain, &p.selection, 100, ROOM, w).unwrap()
    };
    let path = f.root.join("raw.jsonl");
    let (artifact, metrics) =
        evaluation::score(&path, &rows, 5, &mut make(&mut w), &mut w, &f.guard).unwrap();
    assert_eq!((metrics.targets, metrics.raw_correct), (100, 99));
    assert!(metrics.raw99 && metrics.exact99 && metrics.original_success);
    assert!(metrics.ordinary_cross_entropy.unwrap() > 0.08);
    assert_eq!(
        evaluation::replay(&path, &artifact, &rows, 5, &mut make(&mut w), &mut w).unwrap(),
        metrics
    );
    let source = std::fs::read_to_string(&path).unwrap();
    let mut values: Vec<serde_json::Value> = source
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(values[0]["answer"], values[99]["answer"]);
    values[5]["answer"]["numerical"]["logits"][0] = serde_json::json!(123.);
    let corrupted = values.iter().map(|v| format!("{v}\n")).collect::<String>();
    std::fs::write(&path, corrupted).unwrap();
    assert!(
        evaluation::replay(
            &path,
            &disk::hash(&path).unwrap(),
            &rows,
            5,
            &mut make(&mut w),
            &mut w
        )
        .is_err()
    );
    std::fs::write(&path, "x".repeat(1_000_000)).unwrap();
    let mut predictor = make(&mut w);
    let before = w.used;
    assert!(
        evaluation::replay(
            &path,
            &disk::hash(&path).unwrap(),
            &rows,
            5,
            &mut predictor,
            &mut w
        )
        .unwrap_err()
        .to_string()
        .contains("exceeds its forecast")
    );
    assert_eq!(before, w.used);
    rows[98][2] = (rows[98][2] + 1) % 5;
    let (_, metrics) = evaluation::score(
        &f.root.join("98.jsonl"),
        &rows,
        5,
        &mut make(&mut w),
        &mut w,
        &f.guard,
    )
    .unwrap();
    assert_eq!(metrics.raw_correct, 98);
    assert!(!metrics.original_success);
    let mut exhausted = Work::new(1).unwrap();
    let (_, metrics) = evaluation::score(
        &f.root.join("refused.jsonl"),
        &rows,
        5,
        &mut make(&mut w),
        &mut exhausted,
        &f.guard,
    )
    .unwrap();
    assert_eq!(
        (metrics.targets, metrics.refusals, metrics.raw_correct),
        (100, 100, 0)
    );
    assert!(metrics.ordinary_cross_entropy.is_none());
    assert!(!metrics.original_success);
}

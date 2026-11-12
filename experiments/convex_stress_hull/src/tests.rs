use super::*;
use metrics::Predictor;
use std::{
    fs,
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};

fn example(a: i64, b: i64, y: i64) -> data::Example {
    data::Example {
        task: "opaque-test-name".into(),
        tokens: vec![a, b],
        targets: vec![-100, y],
        changes: vec![false, true],
        prompt: vec![],
        answer: vec![],
        limit: 0,
        field: None,
        row_index: None,
    }
}
fn task() -> data::Task {
    data::Task {
        vocab: 2,
        selection: "validation".into(),
        splits: [
            (
                "train".into(),
                vec![example(0, 0, 0), example(0, 1, 1), example(1, 0, 0)],
            ),
            ("validation".into(), vec![example(1, 1, 0)]),
        ]
        .into(),
    }
}
static NEXT: AtomicUsize = AtomicUsize::new(0);
fn monitor(task: &data::Task) -> training::Monitor<'_> {
    let path = std::env::temp_dir().join(format!(
        "convex-hull-unit-{}-{}.jsonl",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let file = fs::File::create(&path).unwrap();
    fs::remove_file(path).unwrap();
    training::Monitor {
        task,
        file,
        started: Instant::now(),
        first99: None,
    }
}
#[test]
fn eligibility_uses_only_training_shape_and_preserves_label_independence() {
    let mut t = task();
    assert!(pair_eligibility(2, &t.splits["train"]).is_none());
    assert_eq!(scale(&t), 1.);
    for e in t.splits.get_mut("train").unwrap() {
        e.targets[1] = 1 - e.targets[1];
        e.task = "different-name".into();
        e.field = Some((99, 123));
    }
    assert!(pair_eligibility(2, &t.splits["train"]).is_none());
    t.splits.get_mut("validation").unwrap()[0].targets[1] = 1;
    assert_eq!(scale(&t), 1.);
    let first = t.splits["train"][0].clone();
    t.splits.get_mut("train").unwrap().push(first);
    assert_eq!(
        pair_eligibility(2, &t.splits["train"]),
        Some("repeated_training_pair".into())
    );
    assert_eq!(scale(&t), 10.);
    t.splits.get_mut("train").unwrap()[0].tokens.push(0);
    assert_eq!(
        pair_eligibility(2, &t.splits["train"]),
        Some("requires_two_tokens_and_one_final_target".into())
    );
}
#[test]
fn scaled_ordinary_ce_matches_the_common_lower_bound() {
    for vocab in [2, 3, 97, 193, 256] {
        for scale in [1., 10.] {
            let mut logits = vec![0.; vocab];
            logits[1] = scale;
            assert!((spectral::loss(&logits, 1) - ideal_ce(vocab, scale)).abs() < 2e-14);
        }
    }
}
#[test]
fn integrated_moment_runner_recovers_the_unseen_binary_control_in_every_mode() -> Result<()> {
    let t = task();
    let mut p = meta::protocol()?;
    for key in [
        "spectral_full_updates",
        "spectral_minibatch_updates",
        "spectral_forward_updates",
    ] {
        p[key] = 1024.into();
    }
    p["frequency_workers"] = 1.into();
    let row = json!({"case":{"batch":2,"observe":2}});
    for arm in ["full_batch", "minibatch", "forward_only"] {
        let out = training::train_moments(arm, "full_rank", &row, &p, &mut monitor(&t))?;
        let model = out.model.unwrap();
        model.validate(&t, "full_rank", 1)?;
        assert!(model.logits(&[1, 1])[0] > 0.99);
        let bytes = serde_json::to_vec(&model)?;
        let restored: Expert = serde_json::from_slice(&bytes)?;
        assert_eq!(restored.logits(&[1, 1]), model.logits(&[1, 1]));
        assert!((teacher_ce(&model, &t.splits["train"])? - ideal_ce(2, 1.)).abs() < 1e-8);
    }
    Ok(())
}
#[test]
fn integrated_count_runner_fits_observations_in_every_mode() -> Result<()> {
    let t = task();
    let p = meta::protocol()?;
    let row = json!({"case":{"batch":2,"observe":2}});
    for arm in ["full_batch", "minibatch", "forward_only"] {
        let out = training::train_counts(arm, &row, &p, &mut monitor(&t))?;
        let model = out.model.unwrap();
        model.validate(&t, "counts", 1)?;
        assert!(teacher_ce(&model, &t.splits["train"])? - ideal_ce(2, 1.) < 1e-8);
        let restored: Expert = serde_json::from_slice(&serde_json::to_vec(&model)?)?;
        assert_eq!(restored.logits(&[1, 1]), model.logits(&[1, 1]));
    }
    Ok(())
}
#[test]
fn selection_prefers_a_training_certificate_and_rejects_impossible_loss() -> Result<()> {
    let ideal = ideal_ce(97, 1.);
    assert_eq!(
        spectral::choose_by_train(&[ideal + 0.1, ideal, ideal], ideal, 1e-8)?,
        1
    );
    assert_eq!(
        spectral::choose_by_train(&[ideal + 0.1, ideal + 0.2], ideal, 1e-8)?,
        0
    );
    assert!(spectral::choose_by_train(&[ideal - 0.01], ideal, 1e-8).is_err());
    Ok(())
}
#[test]
fn encoded_training_keeps_all_supervised_prefixes_and_causal_inputs() {
    let mut t = task();
    t.splits.get_mut("train").unwrap()[0].targets[0] = 1;
    let a = features(&t);
    let (records, ids) = encoded(a, &t.splits["train"]);
    assert_eq!(records.len(), 4);
    assert_eq!(ids[0], vec![0, 1]);
    assert_eq!(records[0].items, a.encode(&[0]));
    assert_eq!(records[1].items, a.encode(&[0, 0]));
    t.splits.get_mut("train").unwrap()[0].tokens[1] = 1;
    let (changed, _) = encoded(a, &t.splits["train"]);
    assert_eq!(changed[0].items, records[0].items);
}

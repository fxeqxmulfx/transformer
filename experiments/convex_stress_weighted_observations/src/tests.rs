use crate::{
    corpus::{Corpus, Limits},
    *,
};
use joint::{
    data::Record,
    head::Head,
    training::{Mode, Policy},
};

// The nontrivial CM57 three-program witness at819d4547, with additional
// repeated observations and a copy row to exercise every feature coordinate.
fn coords() -> Vec<f64> {
    (0..5).map(|c| c as f64 / 4.).collect()
}
fn features(x: usize) -> Vec<Features> {
    (0..3)
        .map(|p| {
            if x < 3 {
                let q = match p {
                    0 => x as f64 / 2.,
                    1 => x as f64 - 1.,
                    _ => 0.,
                };
                [q, 1., 0., 0., 0.]
            } else {
                let mut h = [0.; 5];
                h[x - 1] = 1.;
                h
            }
        })
        .collect()
}
fn rows() -> Vec<Record> {
    [0, 0, 1, 3, 4, 5, 0]
        .into_iter()
        .enumerate()
        .map(|(i, x)| Record {
            id: 100 + i,
            features: features(x),
            label: [1, 2, 3, 0, 4, 4][x],
        })
        .collect()
}
fn corpus(rows: &[Record]) -> Result<Corpus> {
    Corpus::build(
        Limits {
            records: rows.len(),
            groups: rows.len(),
            programs: 3,
            vocabulary: 5,
        },
        rows.iter().cloned().map(Ok),
    )
}
fn theta() -> Vec<Parameters> {
    vec![[2., 1., 0., 4., 4., 4.], [0.5, 1., 0., 2., 2., 2.], [0.; 6]]
}
fn policy(mode: Mode) -> Policy {
    Policy {
        mode,
        epochs: if mode == Mode::Forward { 0 } else { 4 },
        batch: 3,
        learning_rate: 8.,
        initial_total_mass: 12.,
        projection_cycles: 10000,
        projection_tolerance: 1e-10,
    }
}
fn close(a: f64, b: f64, tolerance: f64) {
    assert!((a - b).abs() <= tolerance, "{a} != {b}");
}

#[test]
fn weighted_complete_ce_and_all_derivatives_match_original_rows_and_complex_products() -> Result<()>
{
    let rows = rows();
    let corpus = corpus(&rows)?;
    assert_eq!(corpus.groups().len(), 5);
    let batch = corpus.batch(0..rows.len())?;
    assert_eq!(batch.counts, vec![(0, 3), (1, 1), (2, 1), (3, 1), (4, 1)]);
    let head = Head::new(coords(), theta())?;
    let (loss, grad) = objective::gradient(&corpus, &head, &batch)?;
    let (original, original_grad) = joint::training::gradient(&head, &rows)?;
    close(loss, original, 1e-14);
    close(loss, objective::score(&corpus, &head, &batch)?, 1e-14);
    assert!((loss - joint::training::score(&head, corpus.groups())?).abs() > 1e-4);
    for (a, b) in grad.iter().flatten().zip(original_grad.iter().flatten()) {
        close(*a, *b, 1e-14);
    }
    for p in 0..3 {
        for j in 0..6 {
            let evaluate = |shift: f64| -> Result<f64> {
                let mut t = theta();
                t[p][j] += shift;
                let h = Head::new(coords(), t)?;
                let mut sum = 0.;
                for row in &rows {
                    sum += joint::ce(&h.independent_logits(&row.features)?, row.label);
                }
                Ok(sum / rows.len() as f64)
            };
            close(
                grad[p][j],
                (evaluate(1e-5)? - evaluate(-1e-5)?) / 2e-5,
                5e-9,
            );
        }
    }
    // Equal current logits do not justify merging distinct complete features.
    let zero = Head::new(coords(), vec![[0.; 6]; 3])?;
    assert!(
        corpus
            .groups()
            .iter()
            .all(|r| zero.logits(&r.features).unwrap() == vec![0.; 5])
    );
    assert_eq!(corpus.groups().len(), 5);
    Ok(())
}

#[test]
fn all_modes_match_ungrouped_updates_and_replay_with_original_short_tails() -> Result<()> {
    let rows = rows();
    let corpus = corpus(&rows)?;
    for mode in [Mode::Forward, Mode::Full, Mode::Mini] {
        let weighted = training::train(&corpus, coords(), vec![16.; 3], 128., 1., policy(mode))?;
        let original =
            joint::training::train(&rows, coords(), vec![16.; 3], 128., 1., policy(mode))?;
        training::replay(&weighted, &corpus)?;
        close(weighted.initial_ce, original.initial_ce, 1e-10);
        close(weighted.final_ce, original.final_ce, 1e-10);
        for (a, b) in weighted
            .head
            .theta
            .iter()
            .flatten()
            .zip(original.head.theta.iter().flatten())
        {
            close(*a, *b, 1e-8);
        }
        assert_eq!(weighted.steps.len(), original.steps.len());
        for (a, b) in weighted.steps.iter().zip(&original.steps) {
            let ids: Vec<_> = corpus.occurrences()[a.batch.start..a.batch.end]
                .iter()
                .map(|r| r.id)
                .collect();
            assert_eq!(ids, b.record_ids);
            close(a.before_ce, b.before_ce, 1e-10);
            close(a.after_ce, b.after_ce, 1e-10);
            for (x, y) in a.gradient.iter().flatten().zip(b.gradient.iter().flatten()) {
                close(*x, *y, 1e-10);
            }
        }
        assert_eq!(
            weighted.work.gradient_records,
            original.work.gradient_records
        );
        assert_eq!(
            weighted.work.gradient_batches,
            original.work.gradient_batches
        );
        assert!(weighted.work.complete_logit_products < original.work.complete_logit_products);
        if mode == Mode::Mini {
            assert_eq!(weighted.steps[2].batch, corpus.batch(6..7)?);
            assert_eq!(weighted.steps[2].batch.counts, vec![(0, 1)]);
            assert_eq!(weighted.work.gradient_records, 28);
        }
        if mode == Mode::Forward {
            assert_eq!(weighted.work.gradient_batches, 0);
            assert_eq!(weighted.work.effective_updates, 0);
        } else {
            assert!(weighted.work.effective_updates > 0);
        }
        let encoded = serde_json::to_vec(&weighted)?;
        let mut decoded: training::Outcome = serde_json::from_slice(&encoded)?;
        assert_eq!(decoded, weighted);
        decoded.initial_ce += 1.;
        assert!(training::replay(&decoded, &corpus).is_err());
    }
    Ok(())
}

#[test]
fn each_batch_recounts_repeated_groups_and_weights_the_tail_by_its_actual_size() -> Result<()> {
    let rows = rows();
    let corpus = corpus(&rows)?;
    let head = Head::new(coords(), theta())?;
    let whole = objective::gradient(&corpus, &head, &corpus.batch(0..7)?)?;
    let mut mean = 0.;
    let mut grad = [[0.; 6]; 3];
    for start in (0..7).step_by(3) {
        let batch = corpus.batch(start..(start + 3).min(7))?;
        let (ce, g) = objective::gradient(&corpus, &head, &batch)?;
        let weight = batch.records() as f64 / 7.;
        mean += weight * ce;
        for p in 0..3 {
            for j in 0..6 {
                grad[p][j] += weight * g[p][j];
            }
        }
    }
    close(mean, whole.0, 1e-14);
    for (a, b) in grad.iter().flatten().zip(whole.1.iter().flatten()) {
        close(*a, *b, 1e-14);
    }
    let mut wrong = corpus.batch(6..7)?;
    wrong.counts = corpus.batch(0..7)?.counts;
    assert!(objective::gradient(&corpus, &head, &wrong).is_err());
    assert!(corpus.batch(7..7).is_err());
    assert!(corpus.batch(0..8).is_err());
    Ok(())
}

#[test]
fn a_complete_large_pool_exceeds_the_old_record_limit_without_sampling() -> Result<()> {
    let n = 200_003;
    let corpus = Corpus::build(
        Limits {
            records: n,
            groups: 3,
            programs: 1,
            vocabulary: 5,
        },
        (0..n).map(|i| {
            Ok(Record {
                id: i,
                features: vec![[(i % 3) as f64 / 2., 1., 0., 0., 0.]],
                label: i % 3 + 1,
            })
        }),
    )?;
    assert!(joint::data::forecast(n, 1, 5).is_err());
    assert_eq!(corpus.occurrences().len(), n);
    assert_eq!(corpus.occurrences().last().unwrap().id, n - 1);
    assert_eq!(corpus.groups().len(), 3);
    let whole = corpus.batch(0..n)?;
    assert_eq!(whole.counts, vec![(0, 66668), (1, 66668), (2, 66667)]);
    let mut counted = 0;
    for start in (0..n).step_by(127) {
        let b = corpus.batch(start..(start + 127).min(n))?;
        assert_eq!(
            b.counts.iter().map(|x| x.1).sum::<u64>(),
            b.records() as u64
        );
        counted += b.records();
    }
    assert_eq!(counted, n);
    assert_eq!(corpus.batch(n - n % 127..n)?.records(), 105);
    let mut p = policy(Mode::Full);
    p.epochs = 2;
    let outcome = training::train(&corpus, coords(), vec![16.], 128., 1., p)?;
    assert_eq!(outcome.work.gradient_records, 2 * n);
    assert_eq!(outcome.work.weighted_gradient_groups, 6);
    assert_eq!(
        outcome.work.complete_logit_products,
        16 * 5 * (2 * 3 + 2 * 6)
    );
    training::replay(&outcome, &corpus)?;
    Ok(())
}

#[test]
fn conflicting_noisy_labels_stay_separate_and_exact_anchors_remain_infeasible() -> Result<()> {
    let rows = vec![
        Record {
            id: 0,
            features: features(0),
            label: 1,
        },
        Record {
            id: 1,
            features: features(0),
            label: 2,
        },
        Record {
            id: 2,
            features: features(0),
            label: 1,
        },
    ];
    let corpus = corpus(&rows)?;
    assert_eq!(corpus.groups().len(), 2);
    assert_eq!(corpus.batch(0..3)?.counts, vec![(0, 2), (1, 1)]);
    let head = Head::new(coords(), theta())?;
    close(
        objective::score(&corpus, &head, &corpus.batch(0..3)?)?,
        joint::training::score(&head, &rows)?,
        1e-14,
    );
    let mut p = policy(Mode::Forward);
    p.projection_cycles = 128;
    assert!(training::train(&corpus, coords(), vec![16.; 3], 128., 1., p).is_err());
    Ok(())
}

#[test]
fn bounds_corrupted_counts_labels_ids_and_changed_corpus_bindings_are_rejected() -> Result<()> {
    let rows = rows();
    let c = corpus(&rows)?;
    let encoded = serde_json::to_value(&c)?;
    let mut bad = encoded.clone();
    bad["totals"][0] = serde_json::json!(4);
    assert!(serde_json::from_value::<Corpus>(bad)?.validate().is_err());
    for (field, index, sub, replacement) in [
        ("occurrences", 0, "group", serde_json::json!(100)),
        ("occurrences", 1, "id", serde_json::json!(100)),
        ("groups", 0, "label", serde_json::json!(999)),
        ("groups", 0, "id", serde_json::json!(103)),
    ] {
        let mut bad = encoded.clone();
        bad[field][index][sub] = replacement;
        assert!(serde_json::from_value::<Corpus>(bad)?.validate().is_err());
    }
    let outcome = training::train(&c, coords(), vec![16.; 3], 128., 1., policy(Mode::Forward))?;
    let mut changed = encoded;
    changed["occurrences"][6]["id"] = serde_json::json!(999);
    let changed: Corpus = serde_json::from_value(changed)?;
    changed.validate()?;
    assert!(training::replay(&outcome, &changed).is_err());
    assert!(Corpus::build(c.limits().clone(), rows[..6].iter().cloned().map(Ok)).is_err());
    let mut limits = c.limits().clone();
    limits.groups = 4;
    assert!(Corpus::build(limits, rows.into_iter().map(Ok)).is_err());
    assert!(
        Limits {
            records: 2_000_001,
            groups: 3,
            programs: 1,
            vocabulary: 5
        }
        .buffer_bytes()
        .is_err()
    );
    assert!(
        Limits {
            records: 2_000_000,
            groups: 200_000,
            programs: 256,
            vocabulary: 1024
        }
        .buffer_bytes()
        .is_err()
    );
    Ok(())
}

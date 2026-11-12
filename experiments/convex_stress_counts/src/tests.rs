use super::*;
use metrics::Predictor;

#[test]
fn completion_repairs_a_missing_pair_when_global_sharing_cannot_fit() {
    let mut records = vec![sample(0, 0, 0), sample(0, 2, 2)];
    for q in 1..4 {
        for v in 0..3 {
            records.push(sample(q, v, if q == 3 && v == 0 { 1 } else { v }));
        }
    }
    let refs: Vec<_> = records.iter().enumerate().collect();
    let old_features = convex_stress_repair::Aligned {
        base: features().base,
    };
    let mut old = convex_stress_adaptive::AdaptiveTrainer::new(old_features, 1., records.len());
    let mut new = AdaptiveTrainer::new(features(), 1., records.len());
    old.update_batch(&refs);
    new.update_batch(&refs);
    assert_eq!(old.shared_updates, 0);
    assert_eq!(new.shared_updates, 0);
    assert_eq!(correct(&new.snapshot(), &records), records.len());
    // The chosen slot is lag one: real prefixes [value, query] read that slot.
    assert_eq!(winner(&old.logits(&[1, 0])), 0);
    assert_eq!(winner(&new.logits(&[1, 0])), 1);
    let before = old.objective(&records);
    let after = new.objective(&records);
    assert_eq!(before, after);
    assert!(after.1.abs() < 1e-12);
}

#[test]
fn completion_preserves_observed_rows_even_with_zero_residual() {
    let mut oracle = Oracle {
        selected: 2,
        cost: -3.,
        choices: [(2, vec![(0, 0), (4, 2)])].into(),
    };
    let pooled = Oracle {
        selected: 2,
        cost: -5.,
        choices: [(2, vec![(0, 1), (1, 1), (4, 3)])].into(),
    };
    oracle.complete_missing(&pooled);
    assert_eq!(oracle.cost, -3.);
    assert_eq!(oracle.choices[&2], vec![(0, 0), (1, 1), (4, 2)]);
}

fn features() -> Aligned {
    Aligned {
        base: features::Adapter {
            vocab: 4,
            context: 2,
        },
    }
}
fn sample(q: usize, v: usize, label: usize) -> Observation {
    Observation {
        query: q,
        items: vec![(0, 0), (1, q), (2, v)],
        label,
    }
}
fn winner(z: &[f64]) -> usize {
    z.iter()
        .enumerate()
        .max_by(|(i, a), (j, b)| a.total_cmp(b).then(j.cmp(i)))
        .unwrap()
        .0
}
fn correct(model: &AdaptiveModel, records: &[Observation]) -> usize {
    records
        .iter()
        .filter(|s| winner(&model.raw_logits(s.query, &s.items)) == s.label)
        .count()
}

#[test]
fn full_batch_and_forward_restore_query_specific_xor() {
    let records: Vec<_> = (0..2)
        .flat_map(|q| (0..2).map(move |v| sample(q, v, q ^ v)))
        .collect();
    for scale in [1., 10.] {
        let mut trainer = AdaptiveTrainer::new(features(), scale, records.len());
        trainer.update_batch(&records.iter().enumerate().collect::<Vec<_>>());
        assert_eq!(trainer.shared_updates, 0);
        assert_eq!(correct(&trainer.snapshot(), &records), 4);
        assert!(trainer.objective(&records).1.abs() < 1e-12);
        let mut forward = Forward::new(features(), scale);
        for s in &records {
            forward.update_batch(&[s]);
        }
        let actual = forward.snapshot().unwrap();
        assert_eq!(correct(&actual, &records), 4);
        let mut all = Forward::new(features(), scale);
        all.update_batch(&records.iter().collect::<Vec<_>>());
        assert_eq!(
            serde_json::to_vec(&actual).unwrap(),
            serde_json::to_vec(&all.snapshot().unwrap()).unwrap()
        );
        actual.feasibility_error().unwrap();
    }
}

#[test]
fn shared_tie_retains_the_full_batch_missing_pair_repair() {
    let train = vec![
        sample(0, 0, 1),
        sample(1, 0, 1),
        sample(1, 1, 0),
        sample(2, 1, 0),
    ];
    let all: Vec<_> = (0..3)
        .flat_map(|q| (0..2).map(move |v| sample(q, v, 1 - v)))
        .collect();
    let mut trainer = AdaptiveTrainer::new(features(), 1., train.len());
    trainer.update_batch(&train.iter().enumerate().collect::<Vec<_>>());
    assert_eq!(trainer.shared_updates, 1);
    assert_eq!(correct(&trainer.snapshot(), &all), 6);
    assert!(trainer.objective(&train).1.abs() < 1e-12);
    let mut forward = Forward::new(features(), 1.);
    forward.update_batch(&train.iter().collect::<Vec<_>>());
    assert_eq!(correct(&forward.snapshot().unwrap(), &all), 6);
}

#[test]
fn singleton_memory_survives_optional_sharing_and_replacement() {
    let records = vec![sample(2, 0, 0), sample(2, 1, 1)];
    let mut trainer = AdaptiveTrainer::new(features(), 1., records.len());
    for t in 0..1000 {
        let id = t % 2;
        trainer.update_batch(&[(id, &records[id])]);
    }
    assert_eq!(correct(&trainer.snapshot(), &records), 2);
    assert_eq!(trainer.seen(), 2);
    assert_eq!(trainer.local_gradient[&2].seen, 2);
    assert!(trainer.objective(&records).1 < 1e-5);
    let mut dense_shared = Aggregate::default();
    let mut dense_local = BTreeMap::<usize, Aggregate>::new();
    for saved in trainer.memory.iter().flatten() {
        dense_shared.add(&saved.sample, &saved.residual, true);
        dense_local.entry(saved.sample.query).or_default().add(
            &saved.sample,
            &saved.residual,
            true,
        );
    }
    assert!(
        (dense_shared.oracle(features().storage().pad()).cost
            - trainer
                .shared_gradient
                .oracle(features().storage().pad())
                .cost)
            .abs()
            < 1e-10
    );
    assert!(
        (dense_local[&2].oracle(features().storage().pad()).cost
            - trainer.local_gradient[&2]
                .oracle(features().storage().pad())
                .cost)
            .abs()
            < 1e-10
    );
}

#[test]
fn unrestricted_gap_matches_independent_dense_ce_gradient() {
    let records = vec![
        sample(0, 0, 1),
        sample(0, 1, 0),
        sample(1, 0, 0),
        sample(1, 1, 1),
    ];
    let mut trainer = AdaptiveTrainer::new(features(), 10., records.len());
    for t in 0..7 {
        let i = t % records.len();
        trainer.update_batch(&[(i, &records[i])]);
    }
    assert_dense_gap(&trainer, &records);
}

fn assert_dense_gap(trainer: &AdaptiveTrainer, records: &[Observation]) {
    let a = trainer.model.features.storage();
    let scale = trainer.model.scale;
    let mut gradient = vec![vec![vec![vec![0.; a.vocab]; a.values()]; a.keys()]; a.vocab];
    let (mut ce, mut dot) = (0., 0.);
    for s in records {
        let z = trainer.model.raw_logits(s.query, &s.items);
        let exps: Vec<_> = z.iter().map(|x| (scale * x).exp()).collect();
        let sum: f64 = exps.iter().sum();
        ce += sum.ln() - scale * z[s.label];
        for c in 0..a.vocab {
            let r = scale * (exps[c] / sum - f64::from(c == s.label));
            dot += r * z[c];
            for k in 0..a.keys() {
                gradient[s.query][k][s.value(k, a.pad())][c] += r;
            }
        }
    }
    let oracle: f64 = gradient
        .iter()
        .map(|query| {
            query
                .iter()
                .map(|key| {
                    key.iter()
                        .map(|row| row.iter().copied().fold(f64::INFINITY, f64::min))
                        .sum::<f64>()
                })
                .fold(f64::INFINITY, f64::min)
        })
        .sum();
    let (actual_ce, actual_gap) = trainer.objective(records);
    assert!((actual_ce - ce / records.len() as f64).abs() < 1e-12);
    assert!((actual_gap - (dot - oracle) / records.len() as f64).abs() < 1e-12);
}

#[test]
fn counts_are_causal_preserve_old_slots_and_do_not_add_output_classes() {
    let features = Aligned {
        base: features::Adapter {
            vocab: 6,
            context: 9,
        },
    };
    let old = convex_stress_repair::Aligned {
        base: features.base,
    };
    let layout = features.storage();
    let tokens = [1, 2, 2, 3, 4, 2, 1, 0, 5];
    for n in 1..=tokens.len() {
        let visible = &tokens[..n];
        let items = features.encode(visible);
        assert_eq!(
            items
                .iter()
                .filter(|x| x.0 < layout.old_keys())
                .copied()
                .collect::<Vec<_>>(),
            old.encode(visible)
        );
        assert!(items.windows(2).all(|x| x[0].0 < x[1].0));
        assert!(
            items
                .iter()
                .all(|&(k, v)| k < layout.keys() && v < layout.values())
        );
        let q = visible[n - 1] as usize;
        let record = features.observation(visible, 0);
        for t in 0..layout.vocab {
            let count = visible.iter().filter(|&&x| x as usize == t).count();
            let k = layout.old_keys() + (t + layout.vocab - q) % layout.vocab;
            assert_eq!(record.value(k, layout.pad()), layout.count_value(count));
        }
    }
    let trainer = AdaptiveTrainer::new(features, 1., 1);
    assert_eq!(trainer.logits(&tokens).len(), features.base.vocab);
    assert!(layout.values() > features.base.values());
}

#[test]
fn learned_counts_classify_unseen_permutations_in_all_three_modes() {
    let features = Aligned {
        base: features::Adapter {
            vocab: 6,
            context: 10,
        },
    };
    // Only the test specifies a parity target. Neither features nor trainer
    // receive the rule; each count 0..8 has an ordinary labeled observation.
    let records: Vec<_> = (0..=8)
        .map(|count| {
            let mut tokens = vec![5];
            tokens.extend(std::iter::repeat_n(2, count));
            tokens.extend(std::iter::repeat_n(1, 8 - count));
            tokens.push(4);
            features.observation(&tokens, count % 2)
        })
        .collect();
    let all: Vec<_> = (0..256u32)
        .map(|mask| {
            let mut tokens = vec![5];
            tokens.extend((0..8).map(|i| if mask & (1 << i) == 0 { 1 } else { 2 }));
            tokens.push(4);
            features.observation(&tokens, mask.count_ones() as usize % 2)
        })
        .collect();
    for scale in [1., 10.] {
        let mut full = AdaptiveTrainer::new(features, scale, records.len());
        full.update_batch(&records.iter().enumerate().collect::<Vec<_>>());
        assert_eq!(correct(&full.snapshot(), &all), all.len());
        assert!(full.objective(&records).1.abs() < 1e-11);
        assert_dense_gap(&full, &records);
        let mut mini = AdaptiveTrainer::new(features, scale, records.len());
        for t in 0..300 {
            let i = t % records.len();
            mini.update_batch(&[(i, &records[i])]);
        }
        assert_eq!(correct(&mini.snapshot(), &all), all.len());
        assert_dense_gap(&mini, &records);
        let mut forward = Forward::new(features, scale);
        for s in &records {
            forward.update_batch(&[s]);
        }
        let model = forward.snapshot().unwrap();
        assert_eq!(correct(&model, &all), all.len());
        let mut batch = Forward::new(features, scale);
        batch.update_batch(&records.iter().collect::<Vec<_>>());
        assert_eq!(
            serde_json::to_vec(&model).unwrap(),
            serde_json::to_vec(&batch.snapshot().unwrap()).unwrap()
        );
        model.feasibility_error().unwrap();
    }
}

#[test]
fn integer_joint_snapshots_preserve_predictions_and_domain() {
    let records = [
        sample(0, 0, 0),
        sample(0, 1, 1),
        sample(1, 0, 1),
        sample(1, 1, 0),
    ];
    let mut trainer = AdaptiveTrainer::new(features(), 10., records.len());
    for t in 0..31 {
        let i = t % records.len();
        trainer.update_batch(&[(i, &records[i])]);
        let model = trainer.snapshot();
        model.feasibility_error().unwrap();
        let copy: AdaptiveModel =
            serde_json::from_slice(&serde_json::to_vec(&model).unwrap()).unwrap();
        for q in 0..4 {
            for v in 0..4 {
                let s = sample(q, v, 0);
                let a = model.raw_logits(q, &s.items);
                assert!(a.iter().all(|x| *x >= 0.));
                assert!((a.iter().sum::<f64>() - 1.).abs() < 1e-12);
                assert_eq!(a, copy.raw_logits(q, &s.items));
            }
        }
        assert_eq!(trainer.logits(&[0, 1]), copy.logits(&[0, 1]));
    }
}

#[test]
fn contradictory_data_are_retained_as_forward_failure() {
    let records = vec![sample(0, 1, 0), sample(0, 1, 1)];
    let mut forward = Forward::new(features(), 1.);
    forward.update_batch(&records.iter().collect::<Vec<_>>());
    assert_eq!(forward.snapshot().unwrap_err(), vec![0]);
    let mut trainer = AdaptiveTrainer::new(features(), 1., 2);
    trainer.update_batch(&records.iter().enumerate().collect::<Vec<_>>());
    assert_eq!(correct(&trainer.snapshot(), &records), 1);
}

use super::*;
use metrics::Predictor;

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
    let a = features().storage();
    let mut gradient = vec![vec![vec![vec![0.; a.vocab]; a.values()]; a.keys()]; a.vocab];
    let (mut ce, mut dot) = (0., 0.);
    for s in &records {
        let z = trainer.model.raw_logits(s.query, &s.items);
        let exps: Vec<_> = z.iter().map(|x| (10. * x).exp()).collect();
        let sum: f64 = exps.iter().sum();
        ce += sum.ln() - 10. * z[s.label];
        for c in 0..a.vocab {
            let r = 10. * (exps[c] / sum - f64::from(c == s.label));
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
    let (actual_ce, actual_gap) = trainer.objective(&records);
    assert!((actual_ce - ce / records.len() as f64).abs() < 1e-12);
    assert!((actual_gap - (dot - oracle) / records.len() as f64).abs() < 1e-12);
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

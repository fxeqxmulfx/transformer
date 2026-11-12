use super::*;
use metrics::Predictor;

fn features() -> Aligned {
    Aligned {
        base: features::Adapter {
            vocab: 4,
            context: 3,
        },
    }
}
fn records() -> Vec<Observation> {
    [(0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 1), (0, 0, 2)]
        .into_iter()
        .map(|(q, v, label)| Observation {
            query: q,
            items: vec![(0, 0), (1, q), (2, v)],
            label,
        })
        .collect()
}
fn softmax_ce(z: &[f64], y: usize) -> f64 {
    let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    z.iter().map(|x| (x - max).exp()).sum::<f64>().ln() + max - z[y]
}

#[test]
fn calibrated_softmax_normalization_gradient_and_convexity() {
    let readout = Readout::Calibrated(0.01);
    for q in [[0., 1., 0., 0.], [0.1, 0.2, 0.3, 0.4], [0.25; 4]] {
        let z = readout.logits(&q);
        assert!((z.iter().map(|x| x.exp()).sum::<f64>() - 1.).abs() < 1e-14);
        for y in 0..4 {
            let (_, ce) = readout.residual(&q, y);
            assert!((ce - softmax_ce(&z, y)).abs() < 1e-14);
            assert!(ce >= readout.ideal_ce(4) - 1e-14);
        }
    }
    let q = [0.1, 0.2, 0.3, 0.4];
    let other = [0.4, 0.3, 0.2, 0.1];
    for r in [Readout::Scaled(10.), readout] {
        for y in 0..4 {
            let (g, ce) = r.residual(&q, y);
            let delta = 1e-6;
            let plus: Vec<_> = q
                .iter()
                .zip(other)
                .map(|(x, o)| x + delta * (o - x))
                .collect();
            let minus: Vec<_> = q
                .iter()
                .zip(other)
                .map(|(x, o)| x - delta * (o - x))
                .collect();
            let fd =
                (softmax_ce(&r.logits(&plus), y) - softmax_ce(&r.logits(&minus), y)) / (2. * delta);
            let analytic: f64 = g.iter().map(|(&c, &v)| v * (other[c] - q[c])).sum();
            assert!((fd - analytic).abs() < 1e-8);
            let mid: Vec<_> = q
                .iter()
                .zip(other)
                .map(|(x, o)| 0.3 * x + 0.7 * o)
                .collect();
            assert!(
                softmax_ce(&r.logits(&mid), y)
                    <= 0.3 * ce + 0.7 * softmax_ce(&r.logits(&other), y) + 1e-14
            );
        }
    }
}

#[test]
fn scaled_control_matches_original_count_solver() {
    let records = records();
    let mut old = convex_stress_counts::AdaptiveTrainer::new(features(), 10., records.len());
    let mut new = AdaptiveTrainer::new(features(), Readout::Scaled(10.), records.len());
    for t in 0..9 {
        let id = t % records.len();
        old.update_batch(&[(id, &records[id])]);
        new.update_batch(&[(id, &records[id])]);
        assert_eq!(old.objective(&records), new.objective(&records));
        for a in 0..4 {
            for b in 0..4 {
                assert_eq!(old.logits(&[a, b]), new.logits(&[a, b]));
            }
        }
    }
    assert_eq!(old.shared_updates, new.shared_updates);
    new.snapshot().feasibility_error().unwrap();
}

#[test]
fn calibrated_full_gap_matches_dense_uncentered_oracle() {
    let records = records();
    let mut trainer = AdaptiveTrainer::new(features(), Readout::Calibrated(0.01), records.len());
    for _ in 0..7 {
        trainer.update_batch(&records.iter().enumerate().collect::<Vec<_>>());
    }
    let a = features().storage();
    let mut gradient = vec![vec![vec![vec![0.; a.vocab]; a.values()]; a.keys()]; a.vocab];
    let (mut loss, mut dot) = (0., 0.);
    for s in &records {
        let q = trainer.model.raw_logits(s.query, &s.items);
        let p = 0.99 * q[s.label] + 0.01 / a.vocab as f64;
        loss += -p.ln();
        let derivative = -0.99 / p;
        dot += derivative * q[s.label];
        // Every class, key, value and PAD row is present in the dense oracle.
        for k in 0..a.keys() {
            gradient[s.query][k][s.value(k, a.pad())][s.label] += derivative;
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
    let actual = trainer.objective(&records);
    assert!((actual.0 - loss / records.len() as f64).abs() < 1e-12);
    assert!((actual.1 - (dot - oracle) / records.len() as f64).abs() < 1e-12);
    assert_eq!(actual, trainer.snapshot().objective(&records));
}

#[test]
fn learned_count_rule_generalizes_and_roundtrips() {
    let a = Aligned {
        base: features::Adapter {
            vocab: 4,
            context: 9,
        },
    };
    let train: Vec<_> = (0..=8)
        .map(|ones| {
            let mut tokens = vec![0; 8 - ones];
            tokens.extend(vec![1; ones]);
            tokens.push(3);
            a.observation(&tokens, ones % 2)
        })
        .collect();
    let mut trainer = AdaptiveTrainer::new(a, Readout::Calibrated(0.01), train.len());
    trainer.update_batch(&train.iter().enumerate().collect::<Vec<_>>());
    let model = trainer.snapshot();
    model.feasibility_error().unwrap();
    let bytes = serde_json::to_vec(&model).unwrap();
    let saved: AdaptiveModel = serde_json::from_slice(&bytes).unwrap();
    for code in 0usize..256 {
        let mut tokens: Vec<_> = (0..8).map(|i| ((code >> i) & 1) as i64).collect();
        tokens.push(3);
        assert_eq!(model.logits(&tokens), saved.logits(&tokens));
        assert_eq!(
            channel::argmax(&saved.logits(&tokens)),
            (code.count_ones() % 2) as usize
        );
    }
    let (ce, gap) = model.objective(&train);
    assert!((ce - model.readout.ideal_ce(4)).abs() < 1e-12 && gap.abs() < 1e-12);
    let mut bad = saved;
    bad.readout = Readout::Calibrated(0.);
    assert!(bad.feasibility_error().is_err());
}

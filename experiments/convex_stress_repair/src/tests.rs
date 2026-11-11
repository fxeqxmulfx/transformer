use super::*;
use channel::{argmax, cross_entropy};
use metrics::Predictor;

#[test]
fn memory_repairs_the_frozen_singleton_failure() {
    let base = Adapter {
        vocab: 3,
        context: 2,
    };
    let features = Aligned { base };
    let prefixes = [vec![0, 2], vec![1, 2]];
    let samples: Vec<_> = prefixes
        .iter()
        .enumerate()
        .map(|(label, p)| features.observation(p, label))
        .collect();
    let mut frozen = convex_stress_minibatch::stochastic::Stochastic::new(base);
    let old: Vec<_> = prefixes
        .iter()
        .enumerate()
        .map(|(label, p)| {
            let (query, items) = base.encode(p);
            Observation {
                query,
                items,
                label,
            }
        })
        .collect();
    let mut full = MemoryTrainer::new(features, 1., 2);
    full.update_batch(&samples.iter().enumerate().collect::<Vec<_>>());
    assert!(full.objective(&samples).1.abs() < 1e-14);
    let mut repaired = MemoryTrainer::new(features, 1., 2);
    for t in 0..1000 {
        frozen.update_batch(&[&old[t % 2]]);
        repaired.update_batch(&[(t % 2, &samples[t % 2])]);
        if t >= 1 {
            for (y, p) in prefixes.iter().enumerate() {
                assert_eq!(argmax(&repaired.logits(p)), y);
            }
        }
    }
    assert_eq!(frozen.predict(&prefixes[0]), frozen.predict(&prefixes[1]));
    assert!(frozen.objective(&old.iter().collect::<Vec<_>>()).1 > 0.49);
    let (loss, gap) = repaired.objective(&samples);
    assert!((-1e-12..1e-5).contains(&gap), "gap {gap}");
    assert!(loss - full.objective(&samples).0 < 1e-5);
    assert_eq!(repaired.seen(), 2);
    assert!(repaired.snapshot().channel.feasibility_error().unwrap() < 1e-14);
}

#[test]
fn full_batch_sharing_transfers_a_learned_noncopy_decoder() {
    let base = Adapter {
        vocab: 5,
        context: 3,
    };
    let features = Aligned { base };
    // Query 3 observes both values, identifying a value-dependent route.
    // Queries 2 and 4 each lack one of those values.
    let training = [
        (vec![2, 0, 2], 1),
        (vec![3, 0, 3], 1),
        (vec![3, 1, 3], 0),
        (vec![4, 1, 4], 0),
    ];
    let samples: Vec<_> = training
        .iter()
        .map(|(p, y)| features.observation(p, *y))
        .collect();
    let old: Vec<_> = training
        .iter()
        .map(|(p, y)| {
            let (query, items) = base.encode(p);
            Observation {
                query,
                items,
                label: *y,
            }
        })
        .collect();
    let mut frozen = convex_stress_minibatch::stochastic::Stochastic::new(base);
    frozen.update_batch(&old.iter().collect::<Vec<_>>());
    // All training labels are fitted by the old full-batch learner. Both
    // missing query/value combinations are wrong, despite zero empirical gap.
    for (p, y) in &training {
        assert_eq!(argmax(&frozen.predict(p)), *y);
    }
    assert!(frozen.objective(&old.iter().collect::<Vec<_>>()).1.abs() < 1e-14);
    assert_ne!(argmax(&frozen.predict(&[2, 1, 2])), 0);
    assert_ne!(argmax(&frozen.predict(&[4, 0, 4])), 1);
    let mut model = MemoryTrainer::new(features, 1., samples.len());
    model.update_batch(&samples.iter().enumerate().collect::<Vec<_>>());
    for query in [2, 3, 4] {
        for value in [0, 1] {
            assert_eq!(
                argmax(&model.logits(&[query, value, query])),
                1 - value as usize
            );
        }
    }
    assert!(model.objective(&samples).1.abs() < 1e-14);
}

#[test]
fn sharing_does_not_identify_a_route_on_ambiguous_observations() {
    let features = Aligned {
        base: Adapter {
            vocab: 4,
            context: 3,
        },
    };
    let samples = [
        features.observation(&[2, 0, 2], 1),
        features.observation(&[3, 1, 3], 0),
    ];
    let mut model = MemoryTrainer::new(features, 1., 2);
    model.update_batch(&samples.iter().enumerate().collect::<Vec<_>>());
    assert!(model.objective(&samples).1.abs() < 1e-14);
    // Query-token decoding and inverted value decoding both fit the observations.
    // The fixed first-key rule selects query-token decoding, which is wrong here.
    assert_ne!(argmax(&model.logits(&[2, 1, 2])), 0);
    assert_ne!(argmax(&model.logits(&[3, 0, 3])), 1);
}

fn dense(model: &SharedModel) -> Vec<f64> {
    let a = model.channel.adapter;
    let block = &model.channel.queries[&0];
    let mut out = vec![0.; a.keys() * a.values() * a.vocab];
    for (&k, &mass) in &block.routes {
        for v in 0..a.values() {
            if let Some(row) = block.rows.get(&(k * a.values() + v)) {
                for (&c, &w) in block.classes.iter().zip(row) {
                    out[(k * a.values() + v) * a.vocab + c] = w;
                }
            } else {
                out[(k * a.values() + v) * a.vocab] = mass;
            }
        }
    }
    out
}
fn dense_logits(a: Adapter, weights: &[f64], s: &Observation, scale: f64) -> Vec<f64> {
    (0..a.vocab)
        .map(|c| {
            scale
                * (0..a.keys())
                    .map(|k| weights[(k * a.values() + s.value(k, a.pad())) * a.vocab + c])
                    .sum::<f64>()
        })
        .collect()
}
fn dense_loss(a: Adapter, weights: &[f64], samples: &[Observation], scale: f64) -> f64 {
    samples
        .iter()
        .map(|s| cross_entropy(&dense_logits(a, weights, s, scale), s.label))
        .sum()
}

#[test]
fn fresh_gap_matches_finite_difference_gradient_and_exhaustive_oracle() {
    let features = Aligned {
        base: Adapter {
            vocab: 3,
            context: 3,
        },
    };
    let a = features.storage();
    let samples: Vec<_> = (0..2)
        .flat_map(|x| (0..2).map(move |y| features.observation(&[x, y, 2], (x ^ y) as usize)))
        .collect();
    let mut model = MemoryTrainer::new(features, 10., samples.len());
    model.update_batch(&[(0, &samples[0]), (1, &samples[1])]);
    model.update_batch(&[(2, &samples[2])]);
    let weights = dense(&model.snapshot());
    let mut gradient = vec![0.; weights.len()];
    for s in &samples {
        let z = dense_logits(a, &weights, s, model.scale);
        let denominator: f64 = z.iter().map(|x| x.exp()).sum();
        for k in 0..a.keys() {
            for c in 0..a.vocab {
                gradient[(k * a.values() + s.value(k, a.pad())) * a.vocab + c] +=
                    model.scale * (z[c].exp() / denominator - f64::from(c == s.label));
            }
        }
    }
    for (i, &g) in gradient.iter().enumerate() {
        let mut perturbed = weights.clone();
        perturbed[i] += 1e-6;
        let plus = dense_loss(a, &perturbed, &samples, model.scale);
        perturbed[i] -= 2e-6;
        let minus = dense_loss(a, &perturbed, &samples, model.scale);
        assert!(((plus - minus) / 2e-6 - g).abs() < 2e-8, "coordinate {i}");
    }
    let mut oracle_cost = f64::INFINITY;
    for k in 0..a.keys() {
        for assignment in 0..a.vocab.pow(a.values() as u32) {
            let cost: f64 = (0..a.values())
                .map(|v| {
                    let c = assignment / a.vocab.pow(v as u32) % a.vocab;
                    gradient[(k * a.values() + v) * a.vocab + c]
                })
                .sum();
            oracle_cost = oracle_cost.min(cost);
        }
    }
    let dot: f64 = gradient.iter().zip(&weights).map(|(g, w)| g * w).sum();
    let (loss, gap) = model.objective(&samples);
    assert!((gap - (dot - oracle_cost) / samples.len() as f64).abs() < 1e-12);
    assert!(
        (loss - dense_loss(a, &weights, &samples, model.scale) / samples.len() as f64).abs()
            < 1e-12
    );
}

#[test]
fn memory_replacement_matches_a_dense_sum_of_stored_sample_gradients() {
    let features = Aligned {
        base: Adapter {
            vocab: 3,
            context: 2,
        },
    };
    let a = features.storage();
    let samples = [
        features.observation(&[0, 2], 0),
        features.observation(&[1, 2], 1),
    ];
    let mut model = MemoryTrainer::new(features, 1., 2);
    for t in 0..30 {
        model.update_batch(&[(t % 2, &samples[t % 2])]);
        for k in 0..a.keys() {
            for v in 0..a.values() {
                for c in 0..a.vocab {
                    let expected: f64 = model
                        .memory
                        .iter()
                        .flatten()
                        .filter(|stored| {
                            stored
                                .items
                                .iter()
                                .find(|(key, _)| *key == k)
                                .map(|x| x.1)
                                .unwrap_or(a.pad())
                                == v
                        })
                        .map(|stored| stored.residual.get(&c).copied().unwrap_or(0.))
                        .sum();
                    let actual = if let Some(slot) = model.gradient.keys.get(&k) {
                        if v == a.pad() {
                            model.gradient.total.get(&c).copied().unwrap_or(0.)
                                - slot.totals.get(&c).copied().unwrap_or(0.)
                        } else {
                            slot.rows
                                .get(&v)
                                .and_then(|row| row.get(&c))
                                .copied()
                                .unwrap_or(0.)
                        }
                    } else if v == a.pad() {
                        model.gradient.total.get(&c).copied().unwrap_or(0.)
                    } else {
                        0.
                    };
                    assert!((actual - expected).abs() < 1e-12);
                }
            }
        }
        assert_eq!(model.seen(), (t + 1).min(2));
    }
}

#[test]
fn snapshots_and_saved_models_preserve_live_predictions() {
    let features = Aligned {
        base: Adapter {
            vocab: 4,
            context: 3,
        },
    };
    let prefixes = [vec![2, 0, 2], vec![3, 1, 3], vec![2, 1, 2], vec![3, 0, 3]];
    let mut model = MemoryTrainer::new(features, 10., 4);
    for step in 0..12 {
        let saved: SharedModel =
            serde_json::from_slice(&serde_json::to_vec(&model.snapshot()).unwrap()).unwrap();
        assert!(saved.channel.feasibility_error().unwrap() < 1e-14);
        for p in &prefixes {
            assert_eq!(model.logits(p), saved.logits(p));
        }
        let id = step % 4;
        model.update_batch(&[(
            id,
            &features.observation(&prefixes[id], 1 - prefixes[id][1] as usize),
        )]);
    }
}

#[test]
fn alignment_is_a_causal_permutation_with_no_removed_input_slot() {
    let base = Adapter {
        vocab: 5,
        context: 5,
    };
    let aligned = Aligned { base };
    let prefix = [1, 2, 3, 2];
    let (query, original) = base.encode(&prefix);
    let actual = aligned.encode(&prefix);
    assert_eq!(actual.first(), Some(&(0, 0)));
    assert!(actual.windows(2).all(|w| w[0].0 < w[1].0));
    assert_eq!(actual.len(), original.len() + 1);
    for (k, v) in original {
        let new_key = if k < base.context {
            k + 1
        } else {
            base.context
                + 1
                + (k - base.context) / base.vocab * base.vocab
                + ((k - base.context) % base.vocab + base.vocab - query) % base.vocab
        };
        assert_eq!(
            actual
                .binary_search_by_key(&new_key, |x| x.0)
                .map(|i| actual[i].1),
            Ok(v)
        );
    }
    // Query 2's latest completed successor remains 3; the final query itself
    // does not reveal the next answer.
    assert!(actual.contains(&(base.context + 1, 3)));
    assert_ne!(actual, aligned.encode(&[1, 2, 3, 2, 4]));
}

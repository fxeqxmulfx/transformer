use super::*;
use metrics::Predictor;
use serde_json::{Value, json};
mod modes;

fn request() -> causal::Request {
    causal::Request {
        input: Input {
            vocabulary: 7,
            context: 5,
            observations: vec![
                causal::Observation {
                    tokens: vec![0, 0, 6],
                    label: 2,
                },
                causal::Observation {
                    tokens: vec![0, 1, 6],
                    label: 3,
                },
                causal::Observation {
                    tokens: vec![1, 0, 6],
                    label: 4,
                },
                causal::Observation {
                    tokens: vec![1, 1, 6],
                    label: 5,
                },
            ],
        },
        policy: causal::Policy::default(),
    }
}
fn candidate(x: &causal::Request) -> Result<causal::model::Model> {
    Ok(causal::model::initialize(&causal::discovery::run(x.clone())?)?.model)
}
fn count_value(v: usize, context: usize) -> Value {
    let rows = |weight: u64, shift: usize| {
        (0..v + 2 + context)
            .map(|i| {
                let class = 1 + (i + shift) % (v - 1);
                let mut row = serde_json::Map::new();
                row.insert(class.to_string(), json!(weight));
                Value::Object(row)
            })
            .collect::<Vec<_>>()
    };
    json!({"features":{"base":{"vocab":v,"context":context}},"scale":7.3,
        "shared":{"keys":{"2":{"weight":3,"counts":rows(3,0)}},"active":[2],"total":3},
        "local":{"1":{"keys":{"2":{"weight":2,"counts":rows(2,1)}},"active":[2],"total":2}},"total":7})
}
fn parent(v: usize, context: usize) -> Result<Parent> {
    Parent::from_saved("counts", count_value(v, context))
}
fn winner(z: &[f64]) -> usize {
    (1..z.len()).fold(0, |best, c| if z[c] > z[best] { c } else { best })
}
fn batch_loss(model: &Model, input: &Input, ids: &[usize]) -> Result<f64> {
    let mut total = 0.;
    for &id in ids {
        let s = &input.observations[id];
        let p: Vec<_> = s.tokens.iter().map(|&x| x as i64).collect();
        total += causal::training::ce(&model.logits(&p)?, s.label);
    }
    Ok(total / ids.len() as f64)
}

#[test]
fn frozen_count_complex_product_reproduces_original_bits_and_bounds() -> Result<()> {
    let value = count_value(3, 4);
    let old: convex_stress_counts::AdaptiveModel = serde_json::from_value(value.clone())?;
    old.feasibility_error()?;
    let new = counts::CountParent::from_saved(value)?;
    let bounds = new.validate()?;
    assert!(new.column_l1()?.iter().all(|&n| n <= bounds.radius));
    assert!(bounds.real_slots > bounds.stored_scalars);
    for len in 1..=4 {
        for mut code in 0..3usize.pow(len) {
            let prefix: Vec<_> = (0..len)
                .map(|_| {
                    let t = (code % 3) as i64;
                    code /= 3;
                    t
                })
                .collect();
            let actual = new.complex_logits(&prefix)?;
            let expected = old.logits(&prefix);
            for (z, old) in actual.iter().zip(expected) {
                assert_eq!(z.re.to_bits(), old.to_bits());
                assert_eq!(z.im, 0.);
                assert!(z.norm() <= bounds.radius);
            }
        }
    }
    let empty = json!({"features":{"base":{"vocab":3,"context":4}},"scale":7.3,
        "shared":{"keys":{},"active":[],"total":0},"local":{},"total":0});
    let old: convex_stress_counts::AdaptiveModel = serde_json::from_value(empty.clone())?;
    let zero = counts::CountParent::from_saved(empty)?;
    assert_eq!(
        zero.complex_logits(&[1])?
            .iter()
            .map(|z| z.re)
            .collect::<Vec<_>>(),
        old.logits(&[1])
    );
    Ok(())
}

#[test]
fn actual_route_preserves_parent_outside_observed_support() -> Result<()> {
    let x = request();
    let head = candidate(&x)?;
    let model = Model::new(head.clone(), parent(7, 5)?, &x.input)?;
    let mut known = 0;
    let mut unknown = 0;
    for a in 0..7 {
        for b in 0..7 {
            for q in 0..7 {
                let p = [a, b, q];
                if model.known(&p)? {
                    known += 1;
                    assert_eq!(model.logits(&p)?, head.logits(&p));
                } else {
                    unknown += 1;
                    assert_eq!(model.logits(&p)?, model.parent().logits(&p)?);
                }
            }
        }
    }
    assert!(known > 0 && unknown > 0);
    let a = parent::causal_bounds(&head)?;
    let b = model.parent().bounds()?;
    assert_eq!(model.bounds().radius, a.radius + b.radius);
    assert_eq!(model.bounds().real_slots, a.real_slots + b.real_slots);
    assert_eq!(
        model.bounds().buffer_bytes,
        a.buffer_bytes + b.buffer_bytes + 64 * 12 + 128 * 4
    );
    assert_eq!(
        model.bounds().stored_scalars,
        a.stored_scalars + b.stored_scalars
    );
    // Every feasible candidate update leaves unsupported parent predictions.
    let mut changed = head;
    for w in &mut changed.coefficients {
        *w *= 0.5;
    }
    let updated = Model::new(changed, model.parent().clone(), &x.input)?;
    assert_eq!(updated.logits(&[0, 2, 6])?, model.logits(&[0, 2, 6])?);
    assert!(!model.known(&[0, 2, 6])?);
    Ok(())
}

#[test]
fn full_and_unequal_mini_batches_have_actual_ce_and_finite_difference_gradients() -> Result<()> {
    let mut x = request();
    x.input.observations.push(x.input.observations[0].clone());
    let mut head = candidate(&x)?;
    for w in &mut head.coefficients {
        *w *= 0.02;
    }
    let model = Model::new(head.clone(), parent(7, 5)?, &x.input)?;
    let encoded = head.observations(&x.input)?;
    for ids in [vec![0, 1, 2, 3, 4], vec![0, 1], vec![2, 3], vec![4]] {
        let (loss, gradient) = model.batch_gradient(&x.input, &ids)?;
        let reference = causal::training::gradient(
            &head,
            &ids.iter().map(|&i| encoded[i]).collect::<Vec<_>>(),
        )?;
        assert_eq!((loss, &gradient), (reference.0, &reference.1));
        assert!((loss - batch_loss(&model, &x.input, &ids)?).abs() < 1e-14);
        for index in [0, 7, 14, 21, 35, 49, 63] {
            let epsilon = 1e-5;
            let mut plus = head.clone();
            let mut minus = head.clone();
            plus.coefficients[index] += epsilon;
            minus.coefficients[index] -= epsilon;
            let p = Model::new(plus, model.parent().clone(), &x.input)?;
            let m = Model::new(minus, model.parent().clone(), &x.input)?;
            let actual = (batch_loss(&p, &x.input, &ids)? - batch_loss(&m, &x.input, &ids)?)
                / (2. * epsilon);
            assert!((actual - gradient[index]).abs() < 2e-9, "coordinate{index}");
        }
    }
    assert!(model.batch_gradient(&x.input, &[]).is_err());
    assert!(model.batch_gradient(&x.input, &[0, 0]).is_err());
    assert!(model.batch_gradient(&x.input, &[5]).is_err());
    let mut reordered = x.input;
    reordered.observations.swap(0, 1);
    assert!(model.batch_gradient(&reordered, &[0]).is_err());
    Ok(())
}

#[test]
fn supported_unseen_alias_can_still_regress_with_flat_actual_heads() -> Result<()> {
    let mut x = request();
    x.input.observations = vec![
        x.input.observations[0].clone(),
        causal::Observation {
            tokens: vec![1, 1, 6],
            label: 3,
        },
    ];
    let presence = candidate(&x)?;
    let mut positional = presence.clone();
    positional.blocks[0].decoder.candidate =
        causal::features::Candidate::Single(2 * x.input.vocabulary + 2);
    positional.blocks[0].decoder.labels.swap(0, 1);
    for row in positional.coefficients.as_chunks_mut::<7>().0 {
        row.swap(2, 3);
    }
    positional.validate()?;
    let model = Model::new(presence, Parent::Causal(positional.clone()), &x.input)?;
    let unseen = [0, 1, 6];
    assert!(model.known(&unseen)?);
    assert!(
        x.input
            .observations
            .iter()
            .all(|s| s.tokens != vec![0, 1, 6])
    );
    let target = winner(&positional.logits(&unseen));
    assert_ne!(winner(&model.logits(&unseen)?), target);
    for s in &x.input.observations {
        let p: Vec<_> = s.tokens.iter().map(|&x| x as i64).collect();
        assert!(
            (causal::training::ce(&model.logits(&p)?, s.label)
                - causal::training::ce(&positional.logits(&p), s.label))
            .abs()
                < 1e-14
        );
    }
    Ok(())
}

#[test]
fn saved_state_mutations_cannot_replace_external_bindings() -> Result<()> {
    let x = request();
    let model = Model::new(candidate(&x)?, parent(7, 5)?, &x.input)?;
    let value = serde_json::to_value(&model)?;
    let loaded = Model::load(value.clone(), &x.input, model.binding())?;
    assert_eq!(loaded.logits(&[0, 2, 6])?, model.logits(&[0, 2, 6])?);
    for change in 0..5 {
        let mut bad = value.clone();
        match change {
            0 => bad["candidate"]["coefficients"][0] = json!(0.123),
            1 => bad["parent"]["model"]["scale"] = json!(8.),
            2 => bad["bounds"]["radius"] = json!(1.),
            3 => bad["candidate"]["radius"] = json!(0.),
            _ => {
                bad["parent"]["model"]["scale"] = json!(8.);
                let p: Parent = serde_json::from_value(bad["parent"].clone())?;
                bad["binding"]["parent_sha256"] = json!(digest(&p)?);
            }
        }
        assert!(Model::load(bad, &x.input, model.binding()).is_err());
    }
    let mut changed_input = x.input;
    changed_input.observations[0].label = 0;
    assert!(Model::load(value, &changed_input, model.binding()).is_err());
    Ok(())
}

#[test]
fn unsupported_families_invalid_domains_and_forged_count_states_are_refused() -> Result<()> {
    assert!(Parent::from_saved("prediction_table", json!({})).is_err());
    let original = count_value(7, 5);
    for change in 0..7 {
        let mut bad = original.clone();
        match change {
            0 => bad["total"] = json!((1u64 << 52) + 1),
            1 => bad["shared"]["active"] = json!([]),
            2 => bad["shared"]["keys"]["2"]["counts"][0] = json!({"7":3}),
            3 => bad["shared"]["keys"]["2"]["counts"][0] = json!({"1":4}),
            4 => bad["local"]["1"]["total"] = json!(7),
            5 => bad["scale"] = json!(65.),
            _ => bad["features"]["base"]["context"] = json!(4097),
        }
        assert!(Parent::from_saved("counts", bad).is_err());
    }
    let x = request();
    let model = Model::new(candidate(&x)?, parent(7, 5)?, &x.input)?;
    for p in [vec![], vec![-1], vec![7], vec![0; 6]] {
        assert!(model.logits(&p).is_err());
    }
    assert!(Model::new(candidate(&x)?, parent(7, 4)?, &x.input).is_err());
    let mut extra = x.clone();
    extra.input.observations.push(causal::Observation {
        tokens: vec![0, 0, 5],
        label: 0,
    });
    assert!(Model::new(candidate(&extra)?, parent(7, 5)?, &x.input).is_err());
    let oversized = Input {
        vocabulary: 7,
        context: 5,
        observations: vec![x.input.observations[0].clone(); 200_001],
    };
    assert!(Model::new(candidate(&x)?, parent(7, 5)?, &oversized).is_err());
    Ok(())
}

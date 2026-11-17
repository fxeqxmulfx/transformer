use crate::{
    C, Result,
    model::{Atom, Model},
    oracle::{Gradient, normalize},
    train::{self, Config},
};
use baseline::random::Random;
use bilinear::{
    domain::Shape,
    model::cross_entropy,
    train::{Mode, Record, forward_scalars},
};

mod scaling;
mod support;

fn atom(shape: Shape, seed: u32) -> Result<Atom> {
    let mut rng = Random::new(seed);
    let mut vector = |n| -> Result<Vec<C>> {
        let mut values: Vec<_> = (0..n)
            .map(|_| C::new(rng.gauss(0., 1.), rng.gauss(0., 1.)))
            .collect();
        normalize(&mut values)?;
        Ok(values)
    };
    Ok(Atom {
        output: vector(shape.objects)?,
        query: vector(shape.queries)?,
        context: vector(shape.columns)?,
    })
}

fn model(shape: Shape) -> Result<Model> {
    let mut model = Model::zero(shape, 8.)?;
    model.mix(atom(shape, 17)?, 0.3)?;
    model.mix(atom(shape, 19)?, 0.2)?;
    Ok(model)
}

fn close(a: f64, b: f64, tolerance: f64) {
    assert!((a - b).abs() <= tolerance, "{a} != {b}");
}

#[test]
fn complete_complex_forward_matches_independent_tensor() -> Result<()> {
    let s = Shape::new(3, 2)?;
    let m = model(s)?;
    for length in 0..=3 {
        for mut word in 0..3usize.pow(length as u32) {
            let prefix: Vec<_> = (0..length)
                .map(|_| {
                    let t = (word % 3) as i64;
                    word /= 3;
                    t
                })
                .collect();
            let input = s.addresses(&prefix)?;
            let actual = m.complex(&input);
            for (c, actual) in actual.iter().enumerate() {
                let mut expected = C::default();
                for &q in &input.queries {
                    for &(d, count) in &input.columns {
                        let entry: C = m
                            .atoms
                            .iter()
                            .zip(&m.weights)
                            .map(|(f, &w)| f.entry(c, q, d) * w)
                            .sum();
                        expected += C::new(1., 1.) * entry * count as f64;
                    }
                }
                assert!((*actual - expected).norm() < 2e-14);
            }
            assert!(m.next_object(&prefix)? < s.objects);
        }
    }
    assert!(m.logits(&[0, 1, 2, 0]).is_err());
    assert!(m.logits(&[3]).is_err());
    Ok(())
}

#[test]
fn signed_bias_queries_are_actual_feasible_model_evaluations() -> Result<()> {
    let s = Shape::new(3, 2)?;
    let mut centre = model(s)?;
    let h = 1e-4;
    centre.shrink_for_queries(h)?;
    for prefix in [vec![], vec![1], vec![2, 1, 2]] {
        let base = centre.logits(&prefix)?;
        let measured = forward_scalars(&base, 2, h)?;
        for (c, &measured_scalar) in measured.iter().enumerate() {
            let mut losses = vec![];
            for sign in [1., -1.] {
                let mut query = centre.clone();
                query.weights[0] -= h;
                query.atoms.push(Atom::bias(s, c, sign)?);
                query.weights.push(h);
                query.validate()?;
                let actual = query.logits(&prefix)?;
                for (j, &value) in actual.iter().enumerate() {
                    close(value, base[j] + if j == c { sign * h } else { 0. }, 2e-14);
                }
                losses.push(cross_entropy(&actual, 2)?);
            }
            close(measured_scalar, (losses[0] - losses[1]) / (2. * h), 2e-11);
        }
    }
    Ok(())
}

fn config(mode: Mode) -> Config {
    Config {
        mode,
        query_step: 1e-4,
        momentum: 0.,
        restarts: 2,
        sweeps: 5,
        line_iterations: 24,
    }
}

#[test]
fn full_gradient_retains_both_components_repetitions_and_tail() -> Result<()> {
    let s = Shape::new(3, 4)?;
    let m = model(s)?;
    let records = [
        Record {
            prefix: &[0, 0, 0, 0, 1],
            target: 2,
        },
        Record {
            prefix: &[0, 0, 0, 0, 1],
            target: 2,
        },
        Record {
            prefix: &[2, 1],
            target: 0,
        },
    ];
    for batch in [&records[..], &records[2..]] {
        for mode in [Mode::Mini, Mode::ForwardMini] {
            let mut g = Gradient::zero(s)?;
            train::update_gradient(&mut g, &m, batch, config(mode))?;
            for d in 0..s.columns {
                for q in 0..s.queries {
                    for c in 0..s.objects {
                        for imaginary in [false, true] {
                            let h = 1e-5;
                            let mut difference = 0.;
                            for record in batch {
                                let input = s.addresses(record.prefix)?;
                                let mut base = m.logits(record.prefix)?;
                                let count = input
                                    .columns
                                    .iter()
                                    .find(|&&(i, _)| i == d)
                                    .map_or(0, |&(_, n)| n);
                                let response = f64::from(input.queries.contains(&q))
                                    * count as f64
                                    * if imaginary { -1. } else { 1. };
                                base[c] += h * response;
                                let plus = cross_entropy(&base, record.target)?;
                                base[c] -= 2. * h * response;
                                difference +=
                                    (plus - cross_entropy(&base, record.target)?) / (2. * h);
                            }
                            let expected =
                                g.values[g.index(c, q, d)] * if imaginary { -1. } else { 1. };
                            close(difference / batch.len() as f64, expected, 5e-9);
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn learned_component_search_recovers_a_nonzero_rank_one_control() -> Result<()> {
    let s = Shape::new(3, 1)?;
    let mut g = Gradient::zero(s)?;
    for d in 0..s.columns {
        for q in 0..s.queries {
            for c in 0..s.objects {
                let i = g.index(c, q, d);
                g.values[i] = -(((c + 1) * (q + 1) * (d + 1)) as f64);
            }
        }
    }
    let norm = g.values.iter().map(|x| x * x).sum::<f64>().sqrt();
    let (proposal, value) = g.propose(&mut Random::new(17), 2, 5, None)?;
    proposal.validate(s)?;
    close(value, -2f64.sqrt() * norm, 1e-10);
    assert!(proposal.output.iter().any(|z| z.im != 0.));
    assert!(proposal.query.iter().all(|z| *z != C::default()));
    assert!(proposal.context.iter().all(|z| *z != C::default()));
    Ok(())
}

#[test]
fn both_modes_preserve_full_batches_capacity_and_checkpoint_logits() -> Result<()> {
    let s = Shape::new(3, 2)?;
    let mut endpoints = vec![];
    let records = [
        Record {
            prefix: &[0, 1, 2],
            target: 1,
        },
        Record {
            prefix: &[0, 1, 2],
            target: 1,
        },
        Record {
            prefix: &[2, 2],
            target: 0,
        },
    ];
    for mode in [Mode::Mini, Mode::ForwardMini] {
        let mut m = model(s)?;
        let mut g = Gradient::zero(s)?;
        let mut rng = Random::new(52);
        for batch in [&records[..], &records[2..]] {
            let r = train::step(&mut m, &mut g, batch, config(mode), &mut rng)?;
            assert_eq!(r.records, batch.len());
            assert_eq!(
                r.bias_loss_queries,
                if mode == Mode::Mini {
                    0
                } else {
                    2 * s.objects * batch.len()
                }
            );
            assert!(r.after_ce <= r.centre_ce);
            m.validate()?;
        }
        let saved = serde_json::to_vec(&m)?;
        let restored: Model = serde_json::from_slice(&saved)?;
        restored.validate()?;
        for record in &records {
            assert_eq!(
                m.logits(record.prefix)?
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>(),
                restored
                    .logits(record.prefix)?
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>()
            );
        }
        endpoints.push(m);
    }
    for record in &records {
        for (a, b) in endpoints[0]
            .logits(record.prefix)?
            .iter()
            .zip(endpoints[1].logits(record.prefix)?)
        {
            close(*a, b, 2e-6);
        }
    }
    Ok(())
}

#[test]
fn original_masks_and_full_logit_digests_are_preserved() -> Result<()> {
    let example = baseline::data::Example {
        task: "control".into(),
        tokens: vec![0, 1],
        targets: vec![-100, 0],
        changes: vec![false, true],
        prompt: vec![],
        answer: vec![],
        limit: 0,
        field: None,
        row_index: None,
    };
    let examples = [example.clone(), example];
    let records = bilinear::experiment::records(&examples, &[1, 1, 0])?;
    assert_eq!(records.len(), 3);
    assert!(records.iter().all(|r| r.prefix == [0, 1] && r.target == 0));
    assert!(bilinear::experiment::records(&examples, &[]).is_err());
    let s = Shape::new(2, 1)?;
    let zero = Model::zero(s, 8.)?;
    let mut shifted = zero.clone();
    let mut constant = Atom::zero(s);
    constant.output.fill(C::new(0.5, 0.));
    constant.query[0] = C::new(1., 0.);
    constant.context[0] = C::new(1., 0.);
    shifted.mix(constant, 0.25)?;
    let a = crate::experiment::score::evaluate(&zero, &examples)?;
    let b = crate::experiment::score::evaluate(&shifted, &examples)?;
    assert_eq!(
        a["metrics"]["token_accuracy"],
        b["metrics"]["token_accuracy"]
    );
    assert_eq!(a["forward_calls"], 2);
    assert_ne!(a["raw_logits_sha256"], b["raw_logits_sha256"]);
    Ok(())
}

#[test]
fn scoring_refuses_unsealed_or_changed_protocols() -> Result<()> {
    use crate::experiment::{self, score};
    use bilinear::experiment as original;
    use serde_json::{Value, json};
    let path = std::env::temp_dir().join(format!("shared-atoms-seal-test-{}", std::process::id()));
    std::fs::create_dir(&path)?;
    assert!(score::sealed(&path).is_err());
    let protocol: Value = serde_json::from_str(experiment::PROTOCOL)?;
    let binary = original::hash(&std::fs::read(std::env::current_exe()?)?);
    let mut last = None;
    for case in protocol["cases"].as_array().unwrap() {
        for mode in protocol["modes"].as_array().unwrap() {
            let cell = original::score::cell(
                &path,
                case["name"].as_str().unwrap(),
                mode.as_str().unwrap(),
                1,
            );
            std::fs::create_dir_all(&cell)?;
            original::write_json(
                &cell.join("training_receipt.json"),
                &json!({"status":"sealed",
                "heldout_scored":false,"source_fingerprint":experiment::fingerprint(),"case":case["name"],
                "mode":mode,"seed":1,"updates":case["maximum_updates"]}),
            )?;
            original::write_json(
                &cell.join("metadata.json"),
                &json!({"source_fingerprint":experiment::fingerprint(),
                "protocol":protocol,"case":case,"mode":mode,"seed":1,"binary_sha256":binary}),
            )?;
            if mode != "forward" {
                std::fs::write(cell.join("endpoint.json"), "{}")?;
            }
            last = Some(cell);
        }
    }
    assert!(score::sealed(&path).is_err());
    let last = last.ok_or("no cohort cells")?;
    std::fs::write(last.join("endpoint.json"), "{}")?;
    score::sealed(&path)?;
    let mut meta = original::score::read(&last.join("metadata.json"))?;
    meta["protocol"]["budget_multiplier"] = json!(0);
    std::fs::write(last.join("metadata.json"), serde_json::to_vec(&meta)?)?;
    assert!(score::sealed(&path).is_err());
    std::fs::remove_dir_all(path)?; // Only the temporary fixture created by this test.
    Ok(())
}

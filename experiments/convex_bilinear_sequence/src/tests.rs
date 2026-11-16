use crate::{
    C, algebra,
    domain::Shape,
    model::{Joint, Model, cross_entropy},
    train::{self, Mode, Record},
};
const BUDGET: u64 = 500_000_000;

#[test]
fn frequency_scaling_keeps_full_coordinate_updates_multiplicities_and_tails() {
    use crate::scaling::FrequencyScale;
    let shape = Shape::new(3, 4).unwrap();
    let records = [
        Record {
            prefix: &[0, 1, 0, 1, 0],
            target: 1,
        },
        Record {
            prefix: &[0, 1, 0, 1, 0],
            target: 1,
        },
        Record {
            prefix: &[2, 0],
            target: 2,
        },
    ];
    let scaling = FrequencyScale::from_prefixes(shape, records.iter().map(|r| r.prefix)).unwrap();
    assert_eq!(scaling.records, 3);
    assert_eq!(scaling.value(0, 0).unwrap(), 1.);
    assert_eq!(scaling.value(3, 0).unwrap(), 4.);
    // Pair (0,1) appears twice in each of the two repeated records: 2²+2².
    assert!((scaling.value(0, 1 + 4 * 3 + 1).unwrap() - 4. / 9.).abs() < 1e-15);
    assert!(FrequencyScale::from_prefixes(shape, std::iter::empty()).is_err());
    let model = Model::recover(point(shape), BUDGET).unwrap();
    for mode in [Mode::Mini, Mode::ForwardMini] {
        for batch in [&records[..], &records[2..]] {
            let (plain, plain_receipt) = train::step(&model, batch, mode, 0.01, 1e-4).unwrap();
            let (scaled, receipt) =
                train::step_scaled(&model, batch, mode, 0.01, 1e-4, Some(&scaling)).unwrap();
            assert_eq!(receipt.records, batch.len());
            assert_eq!(receipt.bias_loss_queries, plain_receipt.bias_loss_queries);
            assert!(receipt.diagonal_scaling);
            for q in 0..shape.queries {
                for d in 0..shape.columns {
                    // Independent dense input-response count, without the sparse scale map.
                    let sum: f64 = records
                        .iter()
                        .map(|r| {
                            let a = shape.addresses(r.prefix).unwrap();
                            let response = if a.queries.contains(&q) {
                                a.columns
                                    .iter()
                                    .find(|&&(column, _)| column == d)
                                    .map_or(0., |&(_, n)| n as f64)
                            } else {
                                0.
                            };
                            response * response
                        })
                        .sum();
                    let scale = 4. / (1. + sum);
                    for c in 0..shape.objects {
                        let original = model.joint.entry(c, q, d);
                        let expected = original + (plain.joint.entry(c, q, d) - original) * scale;
                        assert!((scaled.joint.entry(c, q, d) - expected).norm() < 1e-12);
                    }
                }
            }
        }
    }
    let wrong = FrequencyScale::from_prefixes(Shape::new(3, 3).unwrap(), [&[0i64][..]].into_iter())
        .unwrap();
    assert!(train::step_scaled(&model, &records, Mode::Mini, 0.01, 1e-4, Some(&wrong)).is_err());
}

fn example() -> baseline::data::Example {
    baseline::data::Example {
        task: "control".into(),
        tokens: vec![0, 1],
        targets: vec![-100, 0],
        changes: vec![false, true],
        prompt: vec![],
        answer: vec![],
        limit: 0,
        field: None,
        row_index: None,
    }
}

#[test]
fn source_masks_duplicate_examples_and_raw_replay_are_preserved() {
    let examples = [example(), example()];
    let records = crate::experiment::records(&examples, &[1, 1, 0]).unwrap();
    assert_eq!(records.len(), 3);
    assert!(records.iter().all(|r| r.prefix == [0, 1] && r.target == 0));
    assert_eq!(
        crate::experiment::records(&examples, &[0]).unwrap().len(),
        1
    );
    assert!(crate::experiment::records(&examples, &[]).is_err());
    let shape = Shape::new(2, 1).unwrap();
    let zero = Model::recover(Joint::zero(shape), BUDGET).unwrap();
    let mut joint = Joint::zero(shape);
    joint.columns.insert(
        0,
        vec![
            C::new(1., 0.),
            C::default(),
            C::default(),
            C::new(1., 0.),
            C::default(),
            C::default(),
        ],
    );
    let shifted = Model::recover(joint, BUDGET).unwrap();
    let a = crate::experiment::score::evaluate(&zero, &examples).unwrap();
    let b = crate::experiment::score::evaluate(&shifted, &examples).unwrap();
    assert_eq!(
        a["metrics"]["token_accuracy"],
        b["metrics"]["token_accuracy"]
    );
    assert_eq!(a["forward_calls"], 2);
    assert_ne!(a["raw_logits_sha256"], b["raw_logits_sha256"]);
}

#[test]
fn scoring_refuses_an_incomplete_or_changed_cohort_seal() {
    use crate::experiment::{self, score, write_json};
    use serde_json::{Value, json};
    let path = std::env::temp_dir().join(format!("bilinear-seal-test-{}", std::process::id()));
    std::fs::create_dir(&path).unwrap();
    assert!(score::sealed(&path).is_err());
    let protocol: Value = serde_json::from_str(experiment::PROTOCOL).unwrap();
    let mut last = None;
    for config in protocol["cases"].as_array().unwrap() {
        for mode in protocol["modes"].as_array().unwrap() {
            for seed in protocol["seeds"].as_array().unwrap() {
                let cell = score::cell(
                    &path,
                    config["name"].as_str().unwrap(),
                    mode.as_str().unwrap(),
                    seed.as_u64().unwrap(),
                );
                std::fs::create_dir_all(&cell).unwrap();
                let receipt = json!({"status":"sealed","heldout_scored":false,"source_fingerprint":experiment::fingerprint(),
                    "case":config["name"],"mode":mode,"seed":seed,"updates":config["maximum_updates"]});
                write_json(&cell.join("training_receipt.json"), &receipt).unwrap();
                write_json(
                    &cell.join("metadata.json"),
                    &json!({"source_fingerprint":experiment::fingerprint(),
                    "protocol":protocol,"case":config,"mode":mode,"seed":seed}),
                )
                .unwrap();
                last = Some(cell);
                // Deliberately leave one missing endpoint until the next iteration.
                if config != protocol["cases"].as_array().unwrap().last().unwrap()
                    || mode != "forward"
                {
                    std::fs::write(last.as_ref().unwrap().join("endpoint.json"), "{}").unwrap();
                }
            }
        }
    }
    assert!(score::sealed(&path).is_err());
    let last = last.unwrap();
    std::fs::write(last.join("endpoint.json"), "{}").unwrap();
    score::sealed(&path).unwrap();
    let mut receipt = score::read(&last.join("training_receipt.json")).unwrap();
    receipt["updates"] = json!(0);
    std::fs::write(
        last.join("training_receipt.json"),
        serde_json::to_vec(&receipt).unwrap(),
    )
    .unwrap();
    assert!(score::sealed(&path).is_err());
    std::fs::remove_dir_all(path).unwrap();
}

fn point(shape: Shape) -> Joint {
    let mut joint = Joint::zero(shape);
    for d in [0, 1, shape.columns - 1] {
        let column = (0..shape.objects * shape.queries)
            .map(|i| {
                C::new(
                    ((i * 7 + d * 3) % 13) as f64 / 31.,
                    ((i * 3 + d) % 11) as f64 / 29.,
                )
            })
            .collect();
        joint.columns.insert(d, column);
    }
    joint
}

#[test]
fn raw_addresses_retain_pairs_classes_empty_and_longer_prefixes() {
    let s = Shape::new(2, 4).unwrap();
    assert_eq!(s.real_coordinates(), 156);
    let a = s.addresses(&[0, 1, 0, 1, 0]).unwrap();
    assert_eq!(a.queries, [0, 1]);
    assert!(a.columns.contains(&(10, 2)));
    assert!(a.columns.contains(&(11, 1)));
    assert_eq!(s.addresses(&[]).unwrap().columns, [(0, 1)]);
    assert!(s.addresses(&[0; 6]).is_err());
    assert!(s.addresses(&[-1]).is_err());
    assert!(s.addresses(&[2]).is_err());
    assert!(Shape::new(0, 3).is_err());
}

#[test]
fn both_learned_encoders_and_readout_change_as_in_cm86() {
    algebra::set_threads(1).unwrap();
    let s = Shape::new(1, 1).unwrap();
    let mut a = Joint::zero(s);
    a.columns.insert(0, vec![C::new(4., 0.), C::new(0., -1.)]);
    a.columns.insert(1, vec![C::new(0., 1.), C::new(1., 0.)]);
    let mut b = a.clone();
    b.columns.get_mut(&0).unwrap()[0] = C::new(1., 0.);
    b.columns.get_mut(&1).unwrap()[1] = C::new(9., 0.);
    let a = Model::recover(a, BUDGET).unwrap().geometry(0).unwrap();
    let b = Model::recover(b, BUDGET).unwrap().geometry(0).unwrap();
    assert!((a.0 - 5.).abs() < 1e-12 && (b.0 - 10.).abs() < 1e-12);
    for energies in [&a.1, &a.2] {
        assert!((energies[0] - 0.8).abs() < 1e-12);
    }
    for energies in [&b.1, &b.2] {
        assert!((energies[0] - 0.1).abs() < 1e-12);
    }
}

#[test]
fn actual_full_complex_forward_matches_independent_joint_sum_on_all_small_prefixes() {
    let shape = Shape::new(2, 3).unwrap();
    let joint = point(shape);
    let model = Model::recover(joint.clone(), BUDGET).unwrap();
    let mut changed = joint.clone();
    for col in changed.columns.values_mut() {
        for entry in col {
            entry.im = 0.;
        }
    }
    let real_only = Model::recover(changed, BUDGET).unwrap();
    let mut prefixes = vec![vec![]];
    for length in 1..=4 {
        for code in 0..1usize << length {
            prefixes.push((0..length).map(|i| ((code >> i) & 1) as i64).collect());
        }
    }
    let mut imaginary_changes_real_output = false;
    for prefix in prefixes {
        let addresses = shape.addresses(&prefix).unwrap();
        let actual = model.complex(&addresses).unwrap();
        let direct = joint.joint_complex(&addresses);
        assert_eq!(actual.len(), 2);
        for (a, b) in actual.iter().zip(&direct) {
            assert!((*a - *b).norm() < 1e-11);
        }
        let stripped = real_only.logits(&prefix).unwrap();
        imaginary_changes_real_output |= actual
            .iter()
            .zip(stripped)
            .any(|(a, b)| (a.re - b).abs() > 1e-3);
    }
    assert!(imaginary_changes_real_output);
    assert!(model.balance_error < 1e-12);
    assert!(Model::recover(joint, 1).is_err());
}

#[test]
fn all_real_and_imaginary_coordinate_queries_match_forward_only_reuse() {
    let shape = Shape::new(2, 4).unwrap();
    let original = point(shape);
    let prefix = [0, 1, 0, 1, 0];
    let model = Model::recover(original.clone(), BUDGET).unwrap();
    let logits = model.logits(&prefix).unwrap();
    let values = train::forward_scalars(&logits, 1, 1e-3).unwrap();
    let input = shape.addresses(&prefix).unwrap();
    for (c, &difference) in values.iter().enumerate() {
        for q in 0..shape.queries {
            for d in 0..shape.columns {
                for imaginary in [false, true] {
                    let multiplicity = input
                        .columns
                        .iter()
                        .find(|&&(k, _)| k == d)
                        .map_or(0, |&(_, n)| n);
                    let response = if input.queries.contains(&q) {
                        multiplicity as f64
                    } else {
                        0.
                    } * if imaginary { -1. } else { 1. };
                    let h = if response == 0. {
                        1e-3
                    } else {
                        1e-3 / response
                    };
                    let mut losses = vec![];
                    for sign in [1., -1.] {
                        let mut perturbed = original.clone();
                        let column = perturbed
                            .columns
                            .entry(d)
                            .or_insert_with(|| vec![C::default(); shape.objects * shape.queries]);
                        if imaginary {
                            column[c * shape.queries + q].im += sign * h;
                        } else {
                            column[c * shape.queries + q].re += sign * h;
                        }
                        let changed = Model::recover(perturbed, BUDGET).unwrap();
                        losses.push(cross_entropy(&changed.logits(&prefix).unwrap(), 1).unwrap());
                    }
                    let independent = (losses[0] - losses[1]) / (2. * h);
                    assert!(
                        (independent - response * difference).abs() < 1e-8,
                        "class={c} query={q} column={d} imaginary={imaginary}: {independent}"
                    );
                }
            }
        }
    }
}

#[test]
fn mini_and_forward_updates_keep_duplicate_records_and_single_record_tails() {
    let shape = Shape::new(3, 3).unwrap();
    let model = Model::recover(point(shape), BUDGET).unwrap();
    let records = [
        Record {
            prefix: &[0, 1, 2],
            target: 2,
        },
        Record {
            prefix: &[0, 1, 2],
            target: 2,
        },
        Record {
            prefix: &[2, 0],
            target: 0,
        },
    ];
    let (mini, a) = train::step(&model, &records, Mode::Mini, 0.01, 1e-4).unwrap();
    let (forward, b) = train::step(&model, &records, Mode::ForwardMini, 0.01, 1e-4).unwrap();
    assert_eq!(a.records, 3);
    assert_eq!(b.bias_loss_queries, 18);
    assert!(mini.joint != model.joint);
    for c in 0..shape.objects {
        for q in 0..shape.queries {
            for d in 0..shape.columns {
                assert!((mini.joint.entry(c, q, d) - forward.joint.entry(c, q, d)).norm() < 1e-9);
            }
        }
    }
    let (tail, report) =
        train::step(&forward, &records[2..], Mode::ForwardMini, 0.01, 1e-4).unwrap();
    assert_eq!(report.records, 1);
    assert_eq!(report.bias_loss_queries, 6);
    assert!(tail.joint != forward.joint);
    assert!(train::step(&model, &[], Mode::Mini, 0.01, 1e-4).is_err());
    let serialized = serde_json::to_string(&tail.joint).unwrap();
    let replay = Model::recover(serde_json::from_str(&serialized).unwrap(), BUDGET).unwrap();
    assert_eq!(
        tail.logits(&[2, 0]).unwrap(),
        replay.logits(&[2, 0]).unwrap()
    );
}

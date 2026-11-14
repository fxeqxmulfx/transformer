use crate::*;
use convex_stress_shared_roles::training;
use serde_json::json;

fn fixture() -> Result<(kernel::Model, Vec<records::Record>)> {
    let observations = (0..49)
        .filter(|x| ![8, 24, 40].contains(x))
        .map(|x| [x, if x % 7 + x / 7 < 6 { 0 } else { 2 }])
        .collect::<Vec<_>>();
    let report = kernel::discovery::run(kernel::discovery::Request {
        prime: 3,
        input: kernel::discovery::Input {
            n: 7,
            vocabulary: 7,
            observations: observations.clone(),
            policy: kernel::discovery::Policy::default(),
        },
    })?;
    let mut source = kernel::initialize(&report)?.model;
    // Non-saturated logits make the actual gradient test informative.
    for b in &source.blocks {
        for (s, &label) in b.observed_labels.iter().enumerate() {
            source.table[(b.offset + s) * source.vocabulary + label] = 2.;
        }
    }
    source.materialize();
    source.validate()?;
    let mut records = observations
        .iter()
        .enumerate()
        .map(|(id, &[cell, label])| records::Record { id, cell, label })
        .collect::<Vec<_>>();
    records.push(records::Record {
        id: records.len(),
        cell: observations[0][0],
        label: observations[0][1],
    });
    Ok((source, records))
}

fn model() -> Result<Model> {
    let (source, records) = fixture()?;
    let pin = digest(&source)?;
    Model::new(source, records, &pin)
}

#[test]
fn protocol_bank_and_ordinal_distance_ties_are_explicit() -> Result<()> {
    let p: serde_json::Value = serde_json::from_str(PROTOCOL)?;
    assert_eq!(p["neighbors"], json!(BANK));
    assert_eq!(p["canonical_margin"].as_f64(), Some(MARGIN));
    assert_eq!(p["prediction_limit"], json!(PREDICTION_LIMIT));
    let (source, _) = fixture()?;
    let (block, _) = source.location(8);
    let neighbors = nearest(&source, block, 8, 4);
    let cells: Vec<_> = neighbors
        .iter()
        .map(|&s| source.blocks[block].observed[s])
        .collect();
    assert_eq!(cells, vec![1, 7, 9, 15]);
    Ok(())
}

#[test]
fn actual_complex_interpolation_distinguishes_unseen_raw_aliases() -> Result<()> {
    let m = model()?;
    assert!(m.selection.selected_neighbors > 0, "{:?}", m.selection);
    assert_eq!(m.bounds.changed_cells, 3);
    assert_eq!(
        m.source.encoding.code.label(0),
        m.source.encoding.code.label(2)
    );
    for (cell, target) in [(8, 0), (40, 2)] {
        let z = m.logits_cell(cell)?;
        assert!(z[target] > z[2 - target]);
    }
    assert!(m.source.logits_cell(40)[0] > m.source.logits_cell(40)[2]);
    for cell in 0..49 {
        let (block, phi) = m.features(cell)?;
        assert!(
            phi.iter()
                .all(|p| p.re.abs() <= 1. + 1e-12 && p.im.abs() <= 1. + 1e-12)
        );
        if let Some(states) = m.neighbors(cell)? {
            let mut expected = vec![0.; m.source.vocabulary];
            for &s in states {
                for (c, z) in m.source.logits_state(block, s).iter().enumerate() {
                    expected[c] += z / states.len() as f64;
                }
            }
            let actual = m.logits_cell(cell)?;
            for (a, b) in actual.iter().zip(expected) {
                assert!((a - b).abs() < 1e-10);
            }
            for y in 0..m.source.vocabulary {
                let rhs: f64 = states
                    .iter()
                    .map(|&s| {
                        kernel::training::ce(&m.source.logits_state(block, s), y)
                            / states.len() as f64
                    })
                    .sum();
                assert!(kernel::training::ce(&actual, y) <= rhs + 1e-10);
            }
        } else {
            assert_eq!(m.logits_cell(cell)?, m.source.logits_cell(cell));
        }
    }
    Ok(())
}

#[test]
fn all_original_modes_preserve_full_and_repeated_tail_gradients() -> Result<()> {
    let (initial, records) = fixture()?;
    for policy in training::policies(records.len(), 8)? {
        let out = training::run(&initial, &records, policy)?;
        training::replay(&initial, &records, &out)?;
        let pin = digest(&out.model)?;
        let m = Model::new(out.model, records.clone(), &pin)?;
        let encoded = records::encode(m.source(), &records)?;
        let all = (0..records.len()).collect::<Vec<_>>();
        for positions in [&all[..], &all[..8], &all[40..]] {
            let gradient = m.batch_gradient(positions)?;
            let batch = positions.iter().map(|&i| encoded[i]).collect::<Vec<_>>();
            let expected = training::gradient(m.source(), &batch)?;
            assert_eq!(gradient.loss, expected.loss);
            assert_eq!(gradient.coefficients, expected.head);
        }
        for r in &records {
            assert!(m.neighbors(r.cell)?.is_none());
            assert_eq!(m.logits_cell(r.cell)?, m.source.logits_cell(r.cell));
        }
        assert_eq!(
            m.source.coefficients.len(),
            m.bounds.source.real_coefficients
        );
    }
    Ok(())
}

#[test]
fn actual_real_and_imaginary_derivatives_match_finite_differences() -> Result<()> {
    let m = model()?;
    let batch = vec![0, 1, m.records.len() - 1];
    let g = m.batch_gradient(&batch)?;
    let mut checked = [false; 2];
    for (index, &value) in g
        .coefficients
        .iter()
        .enumerate()
        .filter(|(_, v)| v.abs() > 1e-5)
        .take(40)
    {
        let mut plus = m.clone();
        let mut minus = m.clone();
        plus.source.coefficients[index] += 1e-5;
        minus.source.coefficients[index] -= 1e-5;
        let finite =
            (plus.batch_gradient(&batch)?.loss - minus.batch_gradient(&batch)?.loss) / 2e-5;
        assert!((value - finite).abs() < 1e-8);
        checked[(index / m.source.vocabulary) % 2] = true;
    }
    assert_eq!(checked, [true, true]);
    Ok(())
}

#[test]
fn canonical_selection_is_independent_of_trained_coefficients_and_retains_records() -> Result<()> {
    let (mut source, records) = fixture()?;
    let first = selection::select(&source, &records);
    for b in &source.blocks {
        for (s, &label) in b.observed_labels.iter().enumerate() {
            source.table[(b.offset + s) * source.vocabulary + label] = 8.;
        }
    }
    source.materialize();
    source.validate()?;
    assert_eq!(first, selection::select(&source, &records));
    assert!(first.candidates.iter().all(|r| r.records == records.len()));
    for c in first.candidates.iter().skip(1) {
        assert_eq!(c.excluded_state_records, records.len());
    }
    for r in &records {
        let (block, _) = source.location(r.cell);
        let states = nearest(&source, block, r.cell, 9);
        assert!(
            states
                .iter()
                .all(|&s| source.blocks[block].observed[s] != r.cell)
        );
    }
    assert!(
        first.candidates[first
            .candidates
            .iter()
            .position(|c| c.neighbors == first.selected_neighbors)
            .unwrap()]
        .canonical_ce
            < first.candidates[0].canonical_ce
    );
    Ok(())
}

#[test]
fn serialized_state_rejects_weight_selection_head_record_and_budget_changes() -> Result<()> {
    let m = model()?;
    let value = serde_json::to_value(&m)?;
    assert_eq!(m, Model::load(value.clone(), m.binding())?);
    let mut mutations = vec![];
    let mut v = value.clone();
    v["rows"][8] = json!([999]);
    mutations.push(v);
    let mut v = value.clone();
    v["selection"]["selected_neighbors"] = json!(0);
    mutations.push(v);
    let mut v = value.clone();
    v["source"]["coefficients"][0] = json!(999.);
    mutations.push(v);
    let mut v = value.clone();
    v["records"][0]["id"] = json!(12345);
    mutations.push(v);
    let mut v = value.clone();
    v["bounds"]["buffer_bytes"] = json!(0);
    mutations.push(v);
    let mut v = value;
    v["binding"]["protocol_sha256"] = json!("bad");
    mutations.push(v);
    for v in mutations {
        assert!(Model::load(v, m.binding()).is_err());
    }
    Ok(())
}

#[test]
fn unsupported_axes_unique_labels_and_prediction_domain_retain_source_policy() -> Result<()> {
    let report = kernel::discovery::run(kernel::discovery::Request {
        prime: 3,
        input: kernel::discovery::Input {
            n: 3,
            vocabulary: 5,
            observations: vec![[0, 0], [4, 1]],
            policy: kernel::discovery::Policy::default(),
        },
    })?;
    let source = kernel::initialize(&report)?.model;
    let records = vec![
        records::Record {
            id: 0,
            cell: 0,
            label: 0,
        },
        records::Record {
            id: 1,
            cell: 4,
            label: 1,
        },
    ];
    let m = Model::new(source.clone(), records, &digest(&source)?)?;
    assert_eq!(m.selection.selected_neighbors, 0);
    assert_eq!(m.bounds.changed_cells, 0);
    for cell in 0..9 {
        assert_eq!(m.logits_cell(cell)?, source.logits_cell(cell));
    }
    assert!(m.logits(&[0; PREDICTION_LIMIT]).is_ok());
    assert!(m.logits(&vec![0; PREDICTION_LIMIT + 1]).is_err());
    assert!(m.logits(&[]).is_err());
    assert!(m.logits(&[-1, 0]).is_err());
    assert!(m.logits(&[0, 4]).is_err());
    assert!(m.neighbors(9).is_err());
    for positions in [vec![], vec![0, 0], vec![2]] {
        assert!(m.batch_gradient(&positions).is_err());
    }
    Ok(())
}

#[test]
fn original_record_binding_rejects_missing_duplicate_or_wrong_observations() -> Result<()> {
    let (source, records) = fixture()?;
    let pin = digest(&source)?;
    let mut bad = records.clone();
    bad[0].id = bad[1].id;
    assert!(Model::new(source.clone(), bad, &pin).is_err());
    let mut bad = records.clone();
    bad[0].label = 6;
    assert!(Model::new(source.clone(), bad, &pin).is_err());
    let mut bad = records;
    bad.remove(1);
    assert!(Model::new(source.clone(), bad, &pin).is_err());
    assert!(Model::new(source, vec![], &pin).is_err());
    Ok(())
}

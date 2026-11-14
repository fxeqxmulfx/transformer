use crate::*;
use serde_json::json;

fn fixture() -> Result<context::Model> {
    let observations = (0..49)
        .filter(|x| ![8, 24, 40].contains(x))
        .map(|x| {
            [
                x,
                if (x % 7 + x / 7 < 6) ^ [4, 16, 30].contains(&x) {
                    0
                } else {
                    2
                },
            ]
        })
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
    let source = kernel::initialize(&report)?.model;
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
    let pin = digest(&source)?;
    context::Model::new(source, records, &pin)
}
fn model() -> Result<Model> {
    let c = fixture()?;
    let pin = digest(&c)?;
    Model::new(c, &pin)
}
fn winner(z: &[f64]) -> usize {
    z.iter()
        .enumerate()
        .max_by(|(a, x), (b, y)| x.total_cmp(y).then_with(|| b.cmp(a)))
        .unwrap()
        .0
}

#[test]
fn canonical_bank_retains_original_diagnostic_records_and_improves_confidence() -> Result<()> {
    let m = model()?;
    let protocol: serde_json::Value = serde_json::from_str(PROTOCOL)?;
    assert_eq!(protocol["scales"], json!(BANK));
    assert!(m.context().selection().selected_neighbors > 0);
    assert!(m.selection().selected_scale < 1., "{:?}", m.selection());
    let old = &m.selection().candidates[0];
    let selected = m
        .selection()
        .candidates
        .iter()
        .find(|c| c.scale == m.selection().selected_scale)
        .unwrap();
    assert!(selected.canonical_ce + 1e-10 < old.canonical_ce);
    for c in &m.selection().candidates {
        assert_eq!(c.records, m.records().len());
        assert_eq!(c.correct_records, old.correct_records);
        assert_eq!(c.excluded_state_records, old.excluded_state_records);
    }
    assert_eq!(
        m.bounds().context.source.real_coefficients,
        m.source().coefficients.len()
    );
    assert_eq!(m.bounds().scaled_cells, m.context().bounds().changed_cells);
    Ok(())
}

#[test]
fn actual_scaled_complex_products_preserve_raw_ranks_and_bounds() -> Result<()> {
    let mut m = model()?;
    for scale in BANK {
        // Exercise every declared fixed feature scale, not only this fixture's selection.
        m.selection.selected_scale = scale;
        for cell in 0..49 {
            let old = m.context().logits_cell(cell)?;
            let actual = m.logits_cell(cell)?;
            let t = m.scale(cell)?;
            for (a, b) in actual.iter().zip(&old) {
                assert!((a - t * b).abs() < 1e-10);
            }
            assert_eq!(winner(&actual), winner(&old));
            let (_, phi) = m.features(cell)?;
            assert!(
                phi.iter()
                    .all(|p| p.re.abs() <= 1. + 1e-12 && p.im.abs() <= 1. + 1e-12)
            );
            for y in 0..7 {
                let bound = t * kernel::training::ce(&old, y) + (1. - t) * (7f64).ln();
                assert!(kernel::training::ce(&actual, y) <= bound + 1e-10);
            }
        }
    }
    Ok(())
}

#[test]
fn every_mode_keeps_full_repeated_and_tail_objectives_and_gradients() -> Result<()> {
    use convex_stress_shared_roles::training;
    let c = fixture()?;
    for policy in training::policies(c.records().len(), 8)? {
        let outcome = training::run(c.source(), c.records(), policy)?;
        training::replay(c.source(), c.records(), &outcome)?;
        let pin = digest(&outcome.model)?;
        let updated = context::Model::new(outcome.model, c.records().to_vec(), &pin)?;
        let pin = digest(&updated)?;
        let m = Model::new(updated, &pin)?;
        let all = (0..m.records().len()).collect::<Vec<_>>();
        for positions in [&all[..], &all[..8], &all[40..]] {
            let actual = m.batch_gradient(positions)?;
            let expected = m.context().batch_gradient(positions)?;
            assert_eq!(actual.loss, expected.loss);
            assert_eq!(actual.coefficients, expected.coefficients);
            let direct: f64 = positions
                .iter()
                .map(|&i| {
                    let r = &m.records()[i];
                    kernel::training::ce(&m.logits_cell(r.cell).unwrap(), r.label)
                })
                .sum::<f64>()
                / positions.len() as f64;
            assert_eq!(direct, actual.loss);
        }
        for r in m.records() {
            assert_eq!(m.scale(r.cell)?, 1.);
            assert_eq!(m.logits_cell(r.cell)?, m.source().logits_cell(r.cell));
        }
    }
    Ok(())
}

#[test]
fn confidence_selection_does_not_read_trained_coefficients() -> Result<()> {
    let original = fixture()?;
    let first = selection::select(&original)?;
    let mut source = original.source().clone();
    for b in &source.blocks {
        for (s, &label) in b.observed_labels.iter().enumerate() {
            source.table[(b.offset + s) * source.vocabulary + label] = 3.;
        }
    }
    source.materialize();
    source.validate()?;
    let pin = digest(&source)?;
    let other = context::Model::new(source, original.records().to_vec(), &pin)?;
    assert_eq!(first, selection::select(&other)?);
    Ok(())
}

#[test]
fn external_bindings_and_reconstruction_reject_mutated_models() -> Result<()> {
    let m = model()?;
    let value = serde_json::to_value(&m)?;
    assert_eq!(m, Model::load(value.clone(), m.binding())?);
    let paths = [
        vec!["selection", "selected_scale"],
        vec!["bounds", "buffer_bytes"],
        vec!["binding", "context_sha256"],
    ];
    for path in paths {
        let mut bad = value.clone();
        bad[path[0]][path[1]] = json!(0);
        assert!(Model::load(bad, m.binding()).is_err());
    }
    let mut bad = value.clone();
    bad["context"]["source"]["coefficients"][0] = json!(999.);
    assert!(Model::load(bad, m.binding()).is_err());
    let mut bad = value.clone();
    bad["context"]["records"][0]["label"] = json!(6);
    assert!(Model::load(bad, m.binding()).is_err());
    let mut bad = value;
    bad["context"]["rows"][8] = json!([0]);
    assert!(Model::load(bad, m.binding()).is_err());
    assert!(Model::new(m.context().clone(), "bad").is_err());
    Ok(())
}

#[test]
fn unmodified_rows_and_prediction_domain_remain_exact() -> Result<()> {
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
    let pin = digest(&source)?;
    let c = context::Model::new(source, records, &pin)?;
    let pin = digest(&c)?;
    let m = Model::new(c, &pin)?;
    assert_eq!(m.selection().selected_scale, 1.);
    for cell in 0..9 {
        assert_eq!(m.logits_cell(cell)?, m.source().logits_cell(cell));
    }
    assert!(m.logits(&vec![0; 4096]).is_ok());
    for prefix in [vec![], vec![-1, 0], vec![0, 4], vec![0; 4097]] {
        assert!(m.logits(&prefix).is_err());
    }
    for positions in [vec![], vec![0, 0], vec![2]] {
        assert!(m.batch_gradient(&positions).is_err());
    }
    assert!(m.scale(9).is_err());
    Ok(())
}

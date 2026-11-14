use crate::*;
use convex_stress_shared_roles::training;
use serde_json::json;

fn fixture(noisy: bool) -> Result<(kernel::Model, Vec<records::Record>)> {
    fixture_with_holes(noisy, &[8, 24, 40])
}

fn fixture_with_holes(
    noisy: bool,
    holes: &[usize],
) -> Result<(kernel::Model, Vec<records::Record>)> {
    let observations: Vec<_> = (0..49)
        .filter(|x| !holes.contains(x))
        .map(|x| {
            [
                x,
                if (x % 7 + x / 7 < 6) ^ (noisy && [4, 16, 30].contains(&x)) {
                    0
                } else {
                    2
                },
            ]
        })
        .collect();
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
    let mut records: Vec<_> = observations
        .iter()
        .enumerate()
        .map(|(id, &[cell, label])| records::Record { id, cell, label })
        .collect();
    records.push(records::Record {
        id: records.len(),
        cell: observations[0][0],
        label: observations[0][1],
    });
    Ok((source, records))
}

fn wrap(
    source: kernel::Model,
    records: Vec<records::Record>,
    mode: Mode,
    updates: usize,
) -> Result<parent::Model> {
    let ep = Endpoint {
        mode,
        source_reference: "raw-regions-synthetic".into(),
        source_gradient_training_run: mode != Mode::Forward,
        source_optimizer_updates: Some(updates),
        mixture_batch: 8,
    };
    let pin = digest(&source)?;
    let context = parent::scale::context::Model::new(source, records, &pin)?;
    let pin = digest(&context)?;
    let scaled = parent::scale::Model::new(context, &pin)?;
    let pin = digest(&scaled)?;
    let mixture = parent::parent::Model::new(scaled, ep, &pin)?;
    let pin = digest(&mixture)?;
    parent::Model::new(mixture, &pin)
}

fn make(noisy: bool) -> Result<Model> {
    let (source, records) = fixture(noisy)?;
    let parent = wrap(source, records, Mode::Forward, 0)?;
    let pin = digest(&parent)?;
    Model::new(parent, &pin)
}

fn independent_products(m: &Model) -> Result<usize> {
    let mut checked = 0;
    for cell in 0..m.source().n.pow(2) {
        let Some(row) = &m.rows[cell] else {
            assert_eq!(m.logits_cell(cell)?, m.parent.logits_cell(cell)?);
            assert_eq!(m.features(cell)?, m.parent.features(cell)?);
            continue;
        };
        checked += 1;
        let b = &m.source().blocks[row.block];
        let points: Vec<_> = b
            .observed
            .iter()
            .enumerate()
            .map(|(state, &cell)| Point {
                x: (cell % m.source().n) as i64,
                y: (cell / m.source().n) as i64,
                state,
            })
            .collect();
        geometry::verify(
            &row.certificate,
            Point {
                x: (cell % m.source().n) as i64,
                y: (cell / m.source().n) as i64,
                state: usize::MAX,
            },
            &points,
            &b.observed_labels,
            &mut GeometryWork::default(),
        )?;
        let mut expected = vec![0.; m.source().vocabulary];
        let mut jensen = 0.;
        for w in &row.certificate.weights {
            let mass = w.numerator as f64 / row.certificate.denominator as f64;
            // Independent state-space route: complete source products precede
            // interpolation, whereas Model mixes features before the product.
            let logits = m.source().logits_state(row.block, w.state);
            jensen += mass * kernel::training::ce(&logits, row.certificate.label);
            for (a, z) in expected.iter_mut().zip(logits) {
                *a += mass * z;
            }
        }
        let actual = m.logits_cell(cell)?;
        assert_eq!(actual.len(), m.source().vocabulary);
        for (&a, &b) in actual.iter().zip(&expected) {
            assert!((a - b).abs() < 1e-10);
        }
        let winner = (0..actual.len())
            .max_by(|&a, &b| actual[a].total_cmp(&actual[b]).then(b.cmp(&a)))
            .unwrap();
        assert_eq!(winner, row.certificate.label);
        assert!(kernel::training::ce(&actual, row.certificate.label) <= jensen + 1e-10);
        let (block, phi) = m.features(cell)?;
        assert_eq!(block, row.block);
        assert!(
            phi.iter()
                .all(|p| p.re.abs() <= 1. + 1e-12 && p.im.abs() <= 1. + 1e-12)
        );
        assert!(actual.iter().all(|z| z.abs() <= m.source().radius + 1e-10));
    }
    Ok(checked)
}

#[test]
fn all_actual_training_modes_preserve_original_records_gradients_and_tail() -> Result<()> {
    let (source, records) = fixture(false)?;
    assert_eq!(records.len(), 47);
    for policy in training::policies(records.len(), 8)? {
        let mode = match policy.mode {
            training::Mode::Full => Mode::Full,
            training::Mode::Mini => Mode::Mini,
            training::Mode::Forward => Mode::Forward,
        };
        let outcome = training::run(&source, &records, policy)?;
        training::replay(&source, &records, &outcome)?;
        let steps = outcome.steps.len();
        let parent = wrap(outcome.model, records.clone(), mode, steps)?;
        let pin = digest(&parent)?;
        let m = Model::new(parent, &pin)?;
        assert_eq!(m.endpoint().mode, mode);
        assert_eq!(m.endpoint().source_optimizer_updates, Some(steps));
        assert_eq!(m.work.gradient_batches, 0);
        assert_eq!(m.work.optimizer_updates, 0);
        assert_eq!(m.source().coefficients, m.parent.source().coefficients);
        assert_eq!(m.records(), records);
        assert!(
            m.source()
                .blocks
                .iter()
                .any(|b| b.code.iter().any(|p| p.im.abs() > 1e-3))
        );
        assert!(independent_products(&m)? > 0);
        for cell in [8, 24, 40] {
            let label = if cell % 7 + cell / 7 < 6 { 0 } else { 2 };
            let logits = m.logits_cell(cell)?;
            let winner = (0..logits.len())
                .max_by(|&a, &b| logits[a].total_cmp(&logits[b]).then(b.cmp(&a)))
                .unwrap();
            assert_eq!(winner, label);
            let ce = kernel::training::ce(&logits, label);
            assert!(ce < 1e-8);
            if mode == Mode::Forward {
                assert!((ce - (1. + 6. * (-24_f64).exp()).ln()).abs() < 1e-12);
            }
        }
        let s = m.statistics();
        assert_eq!(
            s.observed_cells
                + s.unsupported_cells
                + s.inactive_cells
                + s.overlap_refused_cells
                + s.covered_cells
                + s.uncovered_cells,
            49
        );
        let all: Vec<_> = (0..records.len()).collect();
        for batch in [&all[..], &all[..8], &all[40..]] {
            let old = m.parent.batch_gradient(batch)?;
            let new = m.batch_gradient(batch)?;
            assert_eq!(old.loss, new.loss);
            assert_eq!(old.coefficients, new.coefficients);
            let direct = batch
                .iter()
                .map(|&i| {
                    let r = &m.records()[i];
                    kernel::training::ce(&m.logits_cell(r.cell).unwrap(), r.label)
                })
                .sum::<f64>()
                / batch.len() as f64;
            assert_eq!(direct, new.loss);
        }
        eprintln!(
            "{mode:?}: inherited_updates={steps} covered={} refused_blocks={} new_gradients=0",
            s.covered_cells, s.overlap_refused_blocks
        );
    }
    Ok(())
}

#[test]
fn overlapping_raw_regions_keep_the_entire_original_block() -> Result<()> {
    let m = make(true)?;
    assert!(m.statistics.overlap_refused_blocks > 0);
    assert!(m.statistics.overlap_refused_cells > 0);
    for (cell, row) in m.rows.iter().enumerate() {
        let (block, _) = m.source().location(cell);
        if m.geometry[block].disposition == Disposition::OverlapRefused {
            assert!(row.is_none());
            assert_eq!(m.logits_cell(cell)?, m.parent.logits_cell(cell)?);
        }
    }
    Ok(())
}

#[test]
fn uncovered_inputs_and_unknown_axes_keep_exact_parent_predictions() -> Result<()> {
    let cases = [
        (vec![0, 8, 24, 40], 0, true),
        (
            (0..49).filter(|x| x % 7 == 3 || x / 7 == 3).collect(),
            24,
            false,
        ),
    ];
    for (holes, query, known) in cases {
        let (source, records) = fixture_with_holes(false, &holes)?;
        let parent = wrap(source, records, Mode::Forward, 0)?;
        let pin = digest(&parent)?;
        let model = Model::new(parent, &pin)?;
        assert_eq!(model.source().known(query), known);
        assert!(model.rows[query].is_none());
        assert_eq!(model.logits_cell(query)?, model.parent.logits_cell(query)?);
        assert_eq!(model.features(query)?, model.parent.features(query)?);
        if known {
            let block = model.source().location(query).0;
            assert_eq!(model.geometry[block].disposition, Disposition::Separated);
            assert!(model.statistics.uncovered_cells > 0);
        } else {
            assert!(model.statistics.unsupported_cells > 0);
        }
    }
    Ok(())
}

#[test]
fn multiple_source_blocks_keep_distinct_complex_coefficients() -> Result<()> {
    let observations: Vec<_> = (0..64)
        .filter(|x| ![9, 27, 45, 53].contains(x))
        .map(|x| [x, (x % 8 + x / 8) % 2 + 2 * usize::from(x % 8 + x / 8 < 7)])
        .collect();
    let report = kernel::discovery::run(kernel::discovery::Request {
        prime: 2,
        input: kernel::discovery::Input {
            n: 8,
            vocabulary: 8,
            observations: observations.clone(),
            policy: kernel::discovery::Policy::default(),
        },
    })?;
    let source = kernel::initialize(&report)?.model;
    let records = observations
        .iter()
        .enumerate()
        .map(|(id, &[cell, label])| records::Record { id, cell, label })
        .collect();
    let parent = wrap(source, records, Mode::Forward, 0)?;
    let pin = digest(&parent)?;
    let m = Model::new(parent, &pin)?;
    let blocks: std::collections::BTreeSet<_> = m.rows.iter().flatten().map(|r| r.block).collect();
    assert!(blocks.len() >= 2);
    assert_eq!(independent_products(&m)?, 4);
    Ok(())
}

#[test]
fn saved_geometry_and_rehashed_parent_mutations_are_rejected() -> Result<()> {
    let m = make(false)?;
    let value = serde_json::to_value(&m)?;
    assert_eq!(m, Model::load(value.clone(), m.binding())?);
    let cell = m.rows.iter().position(Option::is_some).unwrap();
    let block = m.rows[cell].as_ref().unwrap().block;
    for pointer in [
        format!("/rows/{cell}/certificate/denominator"),
        format!("/rows/{cell}/certificate/weights/0/numerator"),
        format!("/rows/{cell}/block"),
        format!("/geometry/{block}/hulls/0/vertices/0/x"),
        "/statistics/covered_cells".into(),
        "/work/certificates_verified".into(),
        "/work/gradient_batches".into(),
        "/bounds/buffer_bytes".into(),
    ] {
        let mut bad = value.clone();
        *bad.pointer_mut(&pointer).unwrap() = json!(999);
        assert!(Model::load(bad, m.binding()).is_err(), "{pointer}");
    }
    assert!(Model::new(m.parent.clone(), "wrong").is_err());
    let mut bad = value;
    *bad.pointer_mut("/parent/parent/parent/context/source/coefficients/0")
        .unwrap() = json!(999.);
    let mut rebound = m.binding.clone();
    rebound.parent_sha256 = digest(&bad["parent"])?;
    bad["binding"] = serde_json::to_value(&rebound)?;
    assert!(Model::load(bad, &rebound).is_err());
    Ok(())
}

#[test]
fn bounds_domains_and_fixed_confidence_are_explicit() -> Result<()> {
    let m = make(false)?;
    let b = m.bounds();
    assert!(b.buffer_bytes < BUFFER_LIMIT && b.total_scalar_work < WORK_LIMIT);
    assert!(
        b.prediction_scalar_work_per_query >= 8 * parent::scale::context::PREDICTION_LIMIT as u64
    );
    assert_eq!(b.inherited_scalar_work, m.parent.bounds().total_scalar_work);
    assert_eq!(
        b.total_scalar_work,
        b.inherited_scalar_work + b.construction_scalar_work
    );
    assert!(binding::estimate(m.source(), BUFFER_LIMIT, 0, m.records().len()).is_err());
    assert!(binding::estimate(m.source(), 0, WORK_LIMIT, m.records().len()).is_err());
    let mut bad = m.source().clone();
    bad.n = 258;
    assert!(binding::estimate(&bad, 0, 0, 1).is_err());
    assert!(m.logits(&vec![0; 4096]).is_ok());
    for prefix in [vec![], vec![0], vec![-1, 0], vec![0, 7], vec![0; 4097]] {
        assert!(m.logits(&prefix).is_err());
    }
    assert!(m.logits_cell(49).is_err());
    assert!(m.features(49).is_err());
    for batch in [vec![], vec![0, 0], vec![47]] {
        assert!(m.batch_gradient(&batch).is_err());
    }
    let protocol: serde_json::Value = serde_json::from_str(PROTOCOL)?;
    assert_eq!(protocol["confidence_scale"], 1.);
    assert_eq!(protocol["buffer_limit_bytes"], BUFFER_LIMIT);
    assert_eq!(protocol["scalar_work_limit"], WORK_LIMIT);
    Ok(())
}

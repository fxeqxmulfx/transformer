use crate::*;
use convex_stress_shared_roles::training;
use serde_json::json;

fn fixture(noisy: bool, holes: &[usize]) -> Result<(kernel::Model, Vec<records::Record>)> {
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

fn crossing_fixture() -> Result<(kernel::Model, Vec<records::Record>)> {
    // Two separated triangular regions, with a crossing-only query at (0,3).
    // The observed three-equal/one-different rectangle forces a raw alias in
    // the complete categorical code. The unseen answer is not a constructor input.
    let observations: Vec<_> = [
        (1, 1, 0),
        (1, 5, 0),
        (5, 1, 2),
        (5, 5, 2),
        (3, 1, 0),
        (3, 5, 2),
        (0, 0, 0),
        (6, 6, 2),
        (3, 3, 2),
        (1, 0, 0),
    ]
    .into_iter()
    .map(|(x, y, label)| [x + 7 * y, label])
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
    assert!(source.known(21));
    let records = (0..45)
        .map(|id| {
            let [cell, label] = observations[id % observations.len()];
            records::Record { id, cell, label }
        })
        .collect();
    Ok((source, records))
}

fn wrap(
    source: kernel::Model,
    records: Vec<records::Record>,
    mode: Mode,
    updates: usize,
) -> Result<parent::Model> {
    use regions::parent as gated;
    let ep = Endpoint {
        mode,
        source_reference: "crossing-exclusion-synthetic".into(),
        source_gradient_training_run: mode != Mode::Forward,
        source_optimizer_updates: Some(updates),
        mixture_batch: 8,
    };
    let pin = digest(&source)?;
    let context = gated::scale::context::Model::new(source, records, &pin)?;
    let pin = digest(&context)?;
    let scaled = gated::scale::Model::new(context, &pin)?;
    let pin = digest(&scaled)?;
    let mixture = gated::parent::Model::new(scaled, ep, &pin)?;
    let pin = digest(&mixture)?;
    let g = gated::Model::new(mixture, &pin)?;
    let pin = digest(&g)?;
    let regions = regions::Model::new(g, &pin)?;
    let pin = digest(&regions)?;
    parent::Model::new(regions, &pin)
}

fn make(noisy: bool, holes: &[usize]) -> Result<Model> {
    let (source, records) = fixture(noisy, holes)?;
    let parent = wrap(source, records, Mode::Forward, 0)?;
    let pin = digest(&parent)?;
    Model::new(parent, &pin)
}

fn make_crossing() -> Result<Model> {
    let (source, records) = crossing_fixture()?;
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
            .map(|(i, &cell)| model::point(m.source(), cell, i))
            .collect::<Result<_>>()?;
        exclusion::verify_row(
            row,
            &points,
            &b.observed_labels,
            model::point(m.source(), cell, points.len())?,
            &mut Work::default(),
        )?;
        assert!(m.parent.rows()[cell].is_none());
        let mut expected = vec![0.; m.source().vocabulary];
        let mut jensen = 0.;
        for w in &row.weights {
            let mass = w.numerator as f64 / row.denominator as f64;
            // Independent complete source products precede averaging; the model
            // averages complex source features before the complete head product.
            let logits = m.source().logits_state(row.block, w.state);
            jensen += mass * kernel::training::ce(&logits, row.label);
            for (a, z) in expected.iter_mut().zip(logits) {
                *a += mass * z;
            }
        }
        let actual = m.logits_cell(cell)?;
        for (&a, &b) in actual.iter().zip(&expected) {
            assert!((a - b).abs() < 1e-10);
        }
        let winner = (0..actual.len())
            .max_by(|&a, &b| actual[a].total_cmp(&actual[b]).then(b.cmp(&a)))
            .unwrap();
        assert_eq!(winner, row.label);
        assert!(kernel::training::ce(&actual, row.label) <= jensen + 1e-10);
        let (_, phi) = m.features(cell)?;
        assert!(
            phi.iter()
                .all(|p| p.re.abs() <= 1. + 1e-12 && p.im.abs() <= 1. + 1e-12)
        );
        assert!(actual.iter().all(|z| z.abs() <= m.source().radius + 1e-10));
    }
    Ok(checked)
}

#[test]
fn all_actual_modes_preserve_parent_rows_gradients_repeated_records_and_tail() -> Result<()> {
    let (source, records) = crossing_fixture()?;
    assert_eq!(records.len(), 45);
    for policy in training::policies(records.len(), 8)? {
        let mode = match policy.mode {
            training::Mode::Full => Mode::Full,
            training::Mode::Mini => Mode::Mini,
            training::Mode::Forward => Mode::Forward,
        };
        let outcome = training::run(&source, &records, policy)?;
        training::replay(&source, &records, &outcome)?;
        let steps = outcome.steps.len();
        assert_eq!(
            steps,
            match mode {
                Mode::Full => 1,
                Mode::Mini => 6,
                Mode::Forward => 0,
            }
        );
        assert!(outcome.steps.iter().all(|step| step.head_gradient_l2 > 0.));
        if mode == Mode::Mini {
            assert_eq!(outcome.steps.last().unwrap().count, 5);
            assert_eq!(outcome.work.gradient_examples, 45);
        }
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
        let row = m.rows[21].as_ref().expect("exterior query identified");
        assert_eq!(row.label, 0);
        let ce = kernel::training::ce(&m.logits_cell(21)?, 0);
        assert!(ce < 1e-8);
        if mode == Mode::Forward {
            assert!((ce - (1. + 6. * (-24_f64).exp()).ln()).abs() < 1e-12);
        }
        let s = m.statistics();
        assert_eq!(s.retained_parent_cells + s.eligible_cells, 49);
        assert_eq!(
            s.identified_cells + s.ambiguous_cells + s.inconsistent_cells,
            s.eligible_cells
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
            "{mode:?}: inherited_updates={steps} identified={} new_gradients=0 tail=5",
            s.identified_cells
        );
    }
    Ok(())
}

#[test]
fn overlap_and_unknown_axes_keep_exact_parent() -> Result<()> {
    let noisy = make(true, &[0, 8, 24, 40, 48])?;
    assert!(noisy.parent.parent().statistics().overlap_refused_blocks > 0);
    for (cell, row) in noisy.rows.iter().enumerate() {
        let block = noisy.source().location(cell).0;
        if noisy.parent.parent().geometry()[block].disposition
            == regions::Disposition::OverlapRefused
        {
            assert!(row.is_none());
            assert_eq!(noisy.logits_cell(cell)?, noisy.parent.logits_cell(cell)?);
        }
    }
    let holes: Vec<_> = (0..49).filter(|x| x % 7 == 3 || x / 7 == 3).collect();
    let unknown = make(false, &holes)?;
    assert!(!unknown.source().known(24));
    assert!(unknown.rows[24].is_none());
    assert_eq!(unknown.logits_cell(24)?, unknown.parent.logits_cell(24)?);
    let old = make(false, &[0, 8, 24, 40, 48])?;
    assert!(old.parent.statistics().identified_cells > 0);
    for (cell, row) in old.parent.rows().iter().enumerate() {
        if row.is_some() {
            assert!(old.rows[cell].is_none());
            assert_eq!(old.logits_cell(cell)?, old.parent.logits_cell(cell)?);
        }
    }
    Ok(())
}

#[test]
fn saved_pools_witnesses_predictions_and_cost_mutations_are_rejected() -> Result<()> {
    let m = make_crossing()?;
    let value = serde_json::to_value(&m)?;
    assert_eq!(m, Model::load(value.clone(), m.binding())?);
    let cell = m.rows.iter().position(Option::is_some).unwrap();
    for mutation in 0..11 {
        let mut bad = value.clone();
        match mutation {
            0 => bad["rows"][cell]["pool"] = json!([0]),
            1 => bad["rows"][cell]["exclusions"] = json!([]),
            2 => bad["rows"][cell]["exclusions"][0]["candidate_mixture"]["denominator"] = json!(99),
            3 => {
                bad["rows"][cell]["exclusions"][0]["other_mixture"]["weights"][0]["state"] =
                    json!(999)
            }
            4 => bad["rows"][cell]["weights"][0]["numerator"] = json!(2),
            5 => bad["rows"][cell]["denominator"] = json!(999),
            6 => bad["statistics"]["identified_cells"] = json!(0),
            7 => bad["work"]["candidate_tests"] = json!(0),
            8 => bad["bounds"]["construction_scalar_work"] = json!(0),
            9 => bad["rows"][cell]["block"] = json!(999),
            _ => bad["binding"]["protocol_sha256"] = json!("wrong"),
        }
        assert!(
            Model::load(bad, m.binding()).is_err(),
            "mutation {mutation}"
        );
    }
    assert!(Model::new(m.parent.clone(), "wrong external binding").is_err());
    let mut wrong = m.binding.clone();
    wrong.parent_sha256 = "changed".into();
    assert!(Model::load(value, &wrong).is_err());
    let row = m.rows[cell].as_ref().unwrap();
    model::verify_cell(m.parent(), cell, row, &mut Work::default())?;
    let mut bad_row = row.clone();
    bad_row.block = m.source().blocks.len();
    assert!(model::verify_cell(m.parent(), cell, &bad_row, &mut Work::default()).is_err());
    assert!(
        model::verify_cell(m.parent(), m.source().n.pow(2), row, &mut Work::default()).is_err()
    );
    Ok(())
}

#[test]
fn protocol_bounds_and_input_errors_are_explicit() -> Result<()> {
    let m = make_crossing()?;
    let p: serde_json::Value = serde_json::from_str(PROTOCOL)?;
    for key in [
        "candidate_pool_closure_proved",
        "target_convexity_proved",
        "unconditional_raw_label_identification",
    ] {
        assert_eq!(p[key], false);
    }
    assert_eq!(p["scalar_work_limit"], WORK_LIMIT);
    assert!(m.bounds.buffer_bytes <= BUFFER_LIMIT);
    assert!(m.bounds.total_scalar_work <= WORK_LIMIT);
    assert_eq!(
        m.bounds.total_scalar_work,
        m.bounds.inherited_scalar_work + m.bounds.construction_scalar_work
    );
    assert_eq!(m.bounds, forecast(m.parent())?);
    for prefix in [vec![], vec![1], vec![0, -1], vec![0, 7]] {
        assert!(m.logits(&prefix).is_err());
    }
    assert!(m.logits_cell(49).is_err());
    assert!(m.features(49).is_err());
    Ok(())
}

#[test]
fn resource_limits_and_rehashed_invalid_source_are_enforced() -> Result<()> {
    let m = make_crossing()?;
    assert!(binding::estimate(m.parent(), BUFFER_LIMIT, 0).is_err());
    assert!(binding::estimate(m.parent(), 0, WORK_LIMIT).is_err());
    assert!(m.logits(&vec![0; 4096]).is_ok());
    assert!(m.logits(&vec![0; 4097]).is_err());
    for batch in [vec![], vec![0, 0], vec![45]] {
        assert!(m.batch_gradient(&batch).is_err());
    }
    let mut bad = serde_json::to_value(&m)?;
    *bad.pointer_mut("/parent/parent/parent/parent/parent/context/source/coefficients/0")
        .unwrap() = json!(999.);
    let mut rebound = m.binding.clone();
    rebound.parent_sha256 = digest(&bad["parent"])?;
    bad["binding"] = serde_json::to_value(&rebound)?;
    assert!(Model::load(bad, &rebound).is_err());
    Ok(())
}

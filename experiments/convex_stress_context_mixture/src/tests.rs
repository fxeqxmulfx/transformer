use crate::*;
use serde_json::json;

fn fixture() -> Result<parent::Model> {
    let observations: Vec<_> = (0..49)
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
    wrap(source, records)
}

fn wrap(source: kernel::Model, records: Vec<records::Record>) -> Result<parent::Model> {
    let pin = digest(&source)?;
    let context = parent::context::Model::new(source, records, &pin)?;
    let pin = digest(&context)?;
    parent::Model::new(context, &pin)
}

fn endpoint(mode: Mode) -> Endpoint {
    Endpoint {
        mode,
        source_reference: "synthetic-fixture".into(),
        source_gradient_training_run: mode != Mode::Forward,
        source_optimizer_updates: Some(usize::from(mode != Mode::Forward)),
        mixture_batch: 8,
    }
}

fn make(mode: Mode) -> Result<Model> {
    let p = fixture()?;
    let pin = digest(&p)?;
    Model::new(p, endpoint(mode), &pin)
}

#[test]
fn fixed_bank_and_optimization_keep_original_record_diagnostics() -> Result<()> {
    for mode in [Mode::Full, Mode::Mini, Mode::Forward] {
        let m = make(mode)?;
        let s = m.selection();
        assert_eq!(m.bank().len(), 91);
        assert!(s.canonical_ce <= s.candidates[0].canonical_ce + 1e-10);
        assert!(s.canonical_ce <= s.candidates[s.initial_vertex].canonical_ce + 1e-10);
        assert!(s.weights.iter().all(|a| *a >= 0.));
        assert!((s.weights.iter().sum::<f64>() - 1.).abs() < 1e-12);
        assert!(s.candidates.iter().all(|c| c.records == 47));
        if mode == Mode::Full {
            assert!(s.canonical_ce + 1e-6 < s.candidates[s.initial_vertex].canonical_ce);
            assert!(s.weights.iter().filter(|a| **a > 0.).count() > 1);
        }
        eprintln!(
            "{mode:?}: parent_ce={:.8} vertex_ce={:.8} retained_ce={:.8} correct={}/47 active_weights={}",
            s.candidates[0].canonical_ce,
            s.candidates[s.initial_vertex].canonical_ce,
            s.canonical_ce,
            s.correct_records,
            s.weights.iter().filter(|a| **a > 0.).count()
        );
        let expected = match mode {
            Mode::Full => 64,
            Mode::Mini => 8 * 47usize.div_ceil(8),
            Mode::Forward => 0,
        };
        assert_eq!(s.work.gradient_batches, expected);
        assert_eq!(s.steps.len(), expected);
        assert_eq!(m.bounds.max_gradient_batches, expected);
        assert_eq!(
            s.work.optimizer_updates,
            s.steps.iter().filter(|s| s.gamma > 0.).count()
        );
        if mode == Mode::Forward {
            assert_eq!(s.work.gradient_records, 0);
            assert_eq!(s.work.optimizer_updates, 0);
        }
        if mode == Mode::Mini {
            assert_eq!(s.steps.iter().filter(|s| s.count == 7).count(), 8);
        }
        assert!(s.steps.iter().all(|s| s.after_ce <= s.before_ce + 1e-10));
        assert!(m.bounds.buffer_bytes < 56 * 1024 * 1024 * 1024);
        assert!(m.bounds.total_scalar_work < kernel::limits::WORK_CAP);
    }
    Ok(())
}

#[test]
fn sparse_ce_slopes_match_dense_all_class_loss_and_finite_differences() -> Result<()> {
    let p = fixture()?;
    let bank = bank::build(&p);
    let cache = diagnostic::Cache::new(&p, &bank)?;
    let weights = vec![1. / bank.len() as f64; bank.len()];
    let actual = cache.evaluate(&weights, 0, cache.rows.len(), true);
    let mut dense = 0.;
    for row in &cache.rows {
        let mut z = vec![0.; p.source().vocabulary];
        for (alpha, logits) in weights.iter().zip(&row.logits) {
            for (&c, &value) in row.labels.iter().zip(logits) {
                z[c] += alpha * value;
            }
        }
        dense += kernel::training::ce(&z, row.labels[row.target]);
    }
    assert!((actual.loss - dense / cache.rows.len() as f64).abs() < 1e-12);
    for j in [0, 1, 3, 10, 90] {
        let mut plus = weights.clone();
        let mut minus = weights.clone();
        plus[j] += 1e-5;
        minus[j] -= 1e-5;
        let delta = (cache.evaluate(&plus, 0, 47, false).loss
            - cache.evaluate(&minus, 0, 47, false).loss)
            / 2e-5;
        assert!(
            (delta - actual.slopes[j]).abs() < 1e-7,
            "{j}: {delta} != {}",
            actual.slopes[j]
        );
    }
    let first = cache.evaluate(&weights, 0, 40, true);
    let tail = cache.evaluate(&weights, 40, 7, true);
    assert!((actual.loss - (40. * first.loss + 7. * tail.loss) / 47.).abs() < 1e-12);
    for j in 0..weights.len() {
        assert!(
            (actual.slopes[j] - (40. * first.slopes[j] + 7. * tail.slopes[j]) / 47.).abs() < 1e-12
        );
    }
    Ok(())
}

#[test]
fn canonical_diagnostic_is_the_actual_fixed_complex_initializer() -> Result<()> {
    let p = fixture()?;
    let bank = bank::build(&p);
    let cache = diagnostic::Cache::new(&p, &bank)?;
    assert!(
        p.source()
            .blocks
            .iter()
            .any(|b| b.code.iter().any(|c| c.im.abs() > 1e-3))
    );
    for (&position, j) in [0, 10, 46].iter().zip([0, 17, 90]) {
        let r = &p.records()[position];
        let (block, _) = p.source().location(r.cell);
        let b = &p.source().blocks[block];
        let weights = if bank::ambiguous(b) {
            bank::weights(&bank[j], &bank::distances(p.source(), r.cell))
        } else {
            vec![]
        };
        let mut z = vec![0.; p.source().vocabulary];
        if weights.is_empty() {
            z = p.source().logits_state(block, b.states() - 1);
        } else {
            for w in weights {
                for (a, b) in z.iter_mut().zip(p.source().logits_state(block, w.state)) {
                    *a += w.mass * b;
                }
            }
        }
        let row = &cache.rows[position];
        for (c, &actual) in z.iter().enumerate() {
            let expected = row
                .labels
                .binary_search(&c)
                .map_or(0., |i| row.logits[j][i]);
            assert!((actual - expected).abs() < 1e-10);
        }
    }
    Ok(())
}

#[test]
fn actual_complex_mixture_retains_source_budget_and_jensen_bound() -> Result<()> {
    let mut m = make(Mode::Full)?;
    // A nontrivial simplex exercises the mixture even if selection picks a vertex.
    m.selection.weights.fill(0.);
    m.selection.weights[1] = 0.25;
    m.selection.weights[90] = 0.75;
    m.rows = model::rows(&m.parent, &m.bank, &m.selection.weights)?;
    assert_eq!(m.source().coefficients, m.parent.source().coefficients);
    assert_eq!(m.source().radius, m.parent.source().radius);
    let changed: Vec<_> = m
        .rows
        .iter()
        .enumerate()
        .filter(|(_, row)| row.is_some())
        .map(|(i, _)| i)
        .collect();
    assert!(!changed.is_empty());
    for cell in changed {
        let (block, phi) = m.features(cell)?;
        assert!(
            phi.iter()
                .all(|p| p.re.abs() <= 1. + 1e-12 && p.im.abs() <= 1. + 1e-12)
        );
        let sorted = bank::distances(m.source(), cell);
        let mut bank_logits = Vec::new();
        for j in [1, 90] {
            let mut z = vec![0.; m.source().vocabulary];
            for w in bank::weights(&m.bank[j], &sorted) {
                for (a, b) in z.iter_mut().zip(m.source().logits_state(block, w.state)) {
                    *a += w.mass * b;
                }
            }
            bank_logits.push(z);
        }
        let actual = m.logits_cell(cell)?;
        for (c, z) in actual.iter().enumerate() {
            assert!((z - (0.25 * bank_logits[0][c] + 0.75 * bank_logits[1][c])).abs() < 1e-10);
            assert!(z.abs() <= m.source().radius + 1e-10);
        }
        for y in 0..m.source().vocabulary {
            assert!(
                kernel::training::ce(&actual, y)
                    <= 0.25 * kernel::training::ce(&bank_logits[0], y)
                        + 0.75 * kernel::training::ce(&bank_logits[1], y)
                        + 1e-10
            );
        }
    }
    Ok(())
}

#[test]
fn actual_source_training_modes_keep_observed_repeats_tails_and_gradients() -> Result<()> {
    use convex_stress_shared_roles::training;
    let p = fixture()?;
    for policy in training::policies(p.records().len(), 8)? {
        let mode = match policy.mode {
            training::Mode::Full => Mode::Full,
            training::Mode::Mini => Mode::Mini,
            training::Mode::Forward => Mode::Forward,
        };
        let outcome = training::run(p.source(), p.records(), policy)?;
        training::replay(p.source(), p.records(), &outcome)?;
        let mut ep = endpoint(mode);
        ep.source_optimizer_updates = Some(outcome.steps.len());
        let parent = wrap(outcome.model, p.records().to_vec())?;
        let pin = digest(&parent)?;
        let m = Model::new(parent, ep, &pin)?;
        let all: Vec<_> = (0..m.records().len()).collect();
        for batch in [&all[..], &all[..8], &all[40..]] {
            let old = m.parent.batch_gradient(batch)?;
            let actual = m.batch_gradient(batch)?;
            assert_eq!(old.loss, actual.loss);
            assert_eq!(old.coefficients, actual.coefficients);
            let direct = batch
                .iter()
                .map(|&i| {
                    let r = &m.records()[i];
                    kernel::training::ce(&m.logits_cell(r.cell).unwrap(), r.label)
                })
                .sum::<f64>()
                / batch.len() as f64;
            assert_eq!(direct, actual.loss);
        }
        for r in m.records() {
            assert_eq!(m.logits_cell(r.cell)?, m.parent.logits_cell(r.cell)?);
        }
    }
    Ok(())
}

#[test]
fn canonical_selection_is_independent_of_trained_source_coefficients() -> Result<()> {
    let original = fixture()?;
    let first = selection::select(&original, &bank::build(&original), &endpoint(Mode::Full))?;
    let mut source = original.source().clone();
    for b in &source.blocks {
        for (s, &label) in b.observed_labels.iter().enumerate() {
            source.table[(b.offset + s) * source.vocabulary + label] = 3.;
        }
    }
    source.materialize();
    source.validate()?;
    let other = wrap(source, original.records().to_vec())?;
    assert_eq!(
        first,
        selection::select(&other, &bank::build(&other), &endpoint(Mode::Full))?
    );
    Ok(())
}

#[test]
fn external_bindings_replay_and_reject_mutated_state() -> Result<()> {
    let m = make(Mode::Mini)?;
    let value = serde_json::to_value(&m)?;
    assert_eq!(m, Model::load(value.clone(), m.binding())?);
    for pointer in [
        "/selection/canonical_ce",
        "/selection/weights/0",
        "/bank/0/scale",
        "/bounds/buffer_bytes",
        "/binding/parent_sha256",
        "/endpoint/mixture_batch",
        "/parent/context/source/coefficients/0",
        "/parent/context/records/0/label",
    ] {
        let mut bad = value.clone();
        *bad.pointer_mut(pointer).unwrap() = json!(999);
        assert!(Model::load(bad, m.binding()).is_err(), "{pointer}");
    }
    let cell = m.rows.iter().position(Option::is_some).unwrap();
    let mut bad = value;
    bad["rows"][cell][0]["mass"] = json!(0.);
    assert!(Model::load(bad, m.binding()).is_err());
    assert!(Model::new(m.parent.clone(), endpoint(Mode::Full), "wrong").is_err());
    Ok(())
}

#[test]
fn unchanged_domains_and_invalid_limits_are_explicit() -> Result<()> {
    let m = make(Mode::Forward)?;
    for cell in 0..49 {
        if m.rows[cell].is_none() {
            assert_eq!(m.logits_cell(cell)?, m.parent.logits_cell(cell)?);
        }
    }
    assert!(m.logits(&vec![0; 4096]).is_ok());
    for prefix in [vec![], vec![-1, 0], vec![0, 7], vec![0; 4097]] {
        assert!(m.logits(&prefix).is_err());
    }
    assert!(m.logits_cell(49).is_err());
    assert!(m.features(49).is_err());
    for positions in [vec![], vec![0, 0], vec![47]] {
        assert!(m.batch_gradient(&positions).is_err());
    }
    for batch in [0, usize::MAX] {
        let mut ep = endpoint(Mode::Mini);
        ep.mixture_batch = batch;
        assert!(forecast(m.parent(), &ep).is_err());
    }
    let mut ep = endpoint(Mode::Forward);
    ep.source_gradient_training_run = true;
    assert!(forecast(m.parent(), &ep).is_err());
    let protocol: serde_json::Value = serde_json::from_str(PROTOCOL)?;
    assert_eq!(protocol["line_search"], json!(LINE_SEARCH));
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
    let p = wrap(source, records)?;
    let pin = digest(&p)?;
    let unchanged = Model::new(p, endpoint(Mode::Forward), &pin)?;
    for cell in 0..9 {
        assert_eq!(
            unchanged.logits_cell(cell)?,
            unchanged.source().logits_cell(cell)
        );
    }
    assert!(unchanged.rows.iter().all(Option::is_none));
    Ok(())
}

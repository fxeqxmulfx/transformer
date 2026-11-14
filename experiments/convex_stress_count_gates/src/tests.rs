use crate::*;
use serde_json::json;

fn endpoint(mode: Mode) -> Endpoint {
    Endpoint {
        mode,
        source_reference: "gated-synthetic".into(),
        source_gradient_training_run: mode != Mode::Forward,
        source_optimizer_updates: Some(usize::from(mode != Mode::Forward)),
        mixture_batch: 8,
    }
}

fn wrap_endpoint(
    source: kernel::Model,
    records: Vec<records::Record>,
    ep: Endpoint,
) -> Result<parent::Model> {
    let pin = digest(&source)?;
    let context = scale::context::Model::new(source, records, &pin)?;
    let pin = digest(&context)?;
    let scaled = scale::Model::new(context, &pin)?;
    let pin = digest(&scaled)?;
    parent::Model::new(scaled, ep, &pin)
}

fn wrap(source: kernel::Model, records: Vec<records::Record>, mode: Mode) -> Result<parent::Model> {
    wrap_endpoint(source, records, endpoint(mode))
}

fn fixture(mode: Mode) -> Result<parent::Model> {
    // CM45's noisy ordinal fixture, with a repeated original training record.
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
    wrap(source, records, mode)
}

fn make(mode: Mode) -> Result<Model> {
    let parent = fixture(mode)?;
    let pin = digest(&parent)?;
    Model::new(parent, &pin)
}

#[test]
fn local_optimization_reconstructs_original_modes_and_tail_counts() -> Result<()> {
    for mode in [Mode::Full, Mode::Mini, Mode::Forward] {
        let m = make(mode)?;
        let s = m.selection();
        assert!(s.canonical_ce <= s.initial_ce + 1e-10);
        assert!(s.initial_ce <= m.parent.selection().canonical_ce + 1e-10);
        assert_eq!(m.bank().len(), 91);
        assert_eq!(s.weights.len(), GATE_COUNT);
        assert_eq!(s.anchors.iter().map(|c| c.records).sum::<usize>(), 47);
        assert_eq!(s.excluded_records, 47);
        for (g, row) in s.weights.iter().enumerate() {
            assert!(row.iter().all(|a| *a >= 0.));
            assert!((row.iter().sum::<f64>() - 1.).abs() < 1e-12);
            assert!(
                s.candidates[g]
                    .iter()
                    .all(|c| c.records == s.anchors[g].records)
            );
        }
        let expected = match mode {
            Mode::Full => 64,
            Mode::Mini => 8 * 6,
            Mode::Forward => 0,
        };
        assert_eq!(s.steps.len(), expected);
        assert_eq!(s.work.gradient_batches, expected);
        assert_eq!(m.bounds.max_gradient_batches, expected);
        assert_eq!(
            s.work.optimizer_updates,
            s.steps.iter().filter(|s| s.changed_rows > 0).count()
        );
        assert_eq!(
            s.work.local_row_updates,
            s.steps.iter().map(|s| s.changed_rows).sum::<usize>()
        );
        assert_eq!(
            s.work.gradient_records,
            if mode == Mode::Full {
                64 * 47
            } else if mode == Mode::Mini {
                8 * 47
            } else {
                0
            }
        );
        assert_eq!(
            s.work.score_records,
            47 * 93
                + if mode == Mode::Full {
                    64 * 47 * 7
                } else if mode == Mode::Mini {
                    8 * 47 * 7
                } else {
                    0
                }
        );
        if mode == Mode::Mini {
            assert_eq!(s.steps.iter().filter(|s| s.count == 7).count(), 8);
        }
        if mode == Mode::Forward {
            assert_eq!(s.work.optimizer_updates, 0);
            assert_eq!(s.work.local_row_updates, 0);
        }
        assert!(s.steps.iter().all(|s| s.after_ce <= s.before_ce + 1e-10));
        assert!(m.bounds.buffer_bytes < 56 * 1024 * 1024 * 1024);
        assert!(m.bounds.total_scalar_work < kernel::limits::WORK_CAP);
        assert_eq!(
            m.bounds.inherited_scalar_work,
            m.parent.bounds().total_scalar_work
        );
        eprintln!(
            "{mode:?}: groups={} anchor_ce={:.8} anchor_correct={} local_initial={:.8} final={:.8} correct={}/47 rows_changed={}",
            s.weights.len(),
            m.parent.selection().canonical_ce,
            m.parent.selection().correct_records,
            s.initial_ce,
            s.canonical_ce,
            s.correct_records,
            s.work.local_row_updates
        );
    }
    Ok(())
}

#[test]
fn local_radius_slopes_match_dense_full_vocabulary_and_finite_differences() -> Result<()> {
    let parent = fixture(Mode::Full)?;
    let profiles = gates::profiles(parent.source());
    let gates = gates::build(parent.source(), &profiles);
    let cache = diagnostic::Cache::new(&parent, &gates)?;
    let weights = vec![vec![1. / 91.; 91]; cache.groups];
    let actual = cache.evaluate(&weights, 0, 47, true);
    let dense: f64 = cache
        .rows
        .iter()
        .map(|row| {
            let sparse = row.mixture(&weights[row.gate]);
            let mut z = vec![0.; cache.vocabulary];
            for (&label, &value) in row.labels.iter().zip(&sparse) {
                z[label] = value;
            }
            kernel::training::ce(&z, row.labels[row.target])
        })
        .sum();
    assert!((actual.loss - dense / 47.).abs() < 1e-12);
    for g in 0..cache.groups {
        for j in [0, 1, 17, 90] {
            let mut plus = weights.clone();
            let mut minus = weights.clone();
            plus[g][j] += 1e-5;
            minus[g][j] -= 1e-5;
            let delta = (cache.evaluate(&plus, 0, 47, false).loss
                - cache.evaluate(&minus, 0, 47, false).loss)
                / 2e-5;
            assert!((delta - actual.slopes[g][j]).abs() < 1e-7);
        }
    }
    let first = cache.evaluate(&weights, 0, 40, true);
    let tail = cache.evaluate(&weights, 40, 7, true);
    assert!((actual.loss - (40. * first.loss + 7. * tail.loss) / 47.).abs() < 1e-12);
    for g in 0..cache.groups {
        for j in 0..91 {
            assert!(
                (actual.slopes[g][j] - (40. * first.slopes[g][j] + 7. * tail.slopes[g][j]) / 47.)
                    .abs()
                    < 1e-12
            );
        }
    }
    Ok(())
}

#[test]
fn different_fixed_gates_take_opposite_vertices_and_absent_gates_stay_put() {
    let cache = diagnostic::Cache {
        vocabulary: 4,
        groups: 3,
        bank_size: 2,
        excluded_records: 3,
        rows: vec![
            diagnostic::Row {
                gate: 0,
                anchor: vec![12., 12.],
                labels: vec![0, 2],
                target: 0,
                logits: vec![vec![24., 0.], vec![0., 24.]],
            },
            diagnostic::Row {
                gate: 0,
                anchor: vec![12., 12.],
                labels: vec![0, 2],
                target: 0,
                logits: vec![vec![24., 0.], vec![0., 24.]],
            },
            diagnostic::Row {
                gate: 1,
                anchor: vec![12., 12.],
                labels: vec![0, 2],
                target: 1,
                logits: vec![vec![24., 0.], vec![0., 24.]],
            },
        ],
    };
    let candidates = cache.vertices();
    assert!(candidates[0][0].canonical_ce < candidates[0][1].canonical_ce);
    assert!(candidates[1][1].canonical_ce < candidates[1][0].canonical_ce);
    assert_eq!(candidates[2][0].records, 0);
    let weights = vec![vec![0.5, 0.5]; 3];
    let full = cache.evaluate(&weights, 0, 3, true);
    let tail = cache.evaluate(&weights, 2, 1, true);
    assert_eq!(tail.active, vec![false, true, false]);
    assert_eq!(tail.slopes[0], vec![0., 0.]);
    assert_eq!(tail.slopes[2], vec![0., 0.]);
    assert!(full.slopes[0][0] < full.slopes[0][1]);
    assert!(full.slopes[1][1] < full.slopes[1][0]);
    let mut local = weights.clone();
    assert_eq!(
        selection::update(&mut local, &[None, Some(1), None], 0.5),
        1
    );
    assert_eq!(local[0], weights[0]);
    assert_eq!(local[2], weights[2]);
    assert_eq!(local[1], vec![0.25, 0.75]);
    for (gamma, loss) in
        LINE_SEARCH
            .into_iter()
            .zip(cache.line_scores(&weights, &[None, Some(1), None], 2, 1))
    {
        let mut changed = weights.clone();
        selection::update(&mut changed, &[None, Some(1), None], gamma);
        assert!((loss - cache.evaluate(&changed, 2, 1, false).loss).abs() < 1e-12);
    }
}

#[test]
fn actual_complex_products_equal_anchored_bank_mixture_with_same_budget() -> Result<()> {
    let mut m = make(Mode::Forward)?;
    assert!(
        m.source()
            .blocks
            .iter()
            .any(|b| b.code.iter().any(|c| c.im.abs() > 1e-3))
    );
    for row in &mut m.selection.weights {
        row.fill(0.);
        row[1] = 0.25;
        row[90] = 0.75;
    }
    m.rows = model::rows(&m.parent, &m.selection.weights, &m.gates)?;
    assert_eq!(m.source().coefficients, m.parent.source().coefficients);
    let mut checked = 0;
    for cell in 0..49 {
        if m.rows[cell].is_none() {
            continue;
        }
        checked += 1;
        let (block, phi) = m.features(cell)?;
        assert_eq!(block, m.source().location(cell).0);
        assert!(
            phi.iter()
                .all(|p| p.re.abs() <= 1. + 1e-12 && p.im.abs() <= 1. + 1e-12)
        );
        let effective = model::effective(&m.anchor, &m.selection.weights[m.gates[cell]]);
        let sorted = bank::distances(m.source(), cell);
        let mut expected = vec![0.; m.source().vocabulary];
        let mut jensen = vec![0.; m.source().vocabulary];
        for (&alpha, context) in effective.iter().zip(m.bank()) {
            let mut z = vec![0.; expected.len()];
            for w in bank::weights(context, &sorted) {
                for (a, b) in z.iter_mut().zip(m.source().logits_state(block, w.state)) {
                    *a += w.mass * b;
                }
            }
            for (y, loss) in jensen.iter_mut().enumerate() {
                *loss += alpha * kernel::training::ce(&z, y);
            }
            for (value, raw) in expected.iter_mut().zip(z) {
                *value += alpha * raw;
            }
        }
        let actual = m.logits_cell(cell)?;
        for (y, (&a, &b)) in actual.iter().zip(&expected).enumerate() {
            assert!((a - b).abs() < 1e-10);
            assert!(a.abs() <= m.source().radius + 1e-10);
            assert!(kernel::training::ce(&actual, y) <= jensen[y] + 1e-10);
        }
    }
    assert!(checked > 0);
    Ok(())
}

#[test]
fn every_actual_training_mode_keeps_observed_head_loss_and_derivatives() -> Result<()> {
    use convex_stress_shared_roles::training;
    let fixture = fixture(Mode::Forward)?;
    for policy in training::policies(47, 8)? {
        let mode = match policy.mode {
            training::Mode::Full => Mode::Full,
            training::Mode::Mini => Mode::Mini,
            training::Mode::Forward => Mode::Forward,
        };
        let outcome = training::run(fixture.source(), fixture.records(), policy)?;
        training::replay(fixture.source(), fixture.records(), &outcome)?;
        let mut ep = endpoint(mode);
        ep.source_optimizer_updates = Some(outcome.steps.len());
        let parent = wrap_endpoint(outcome.model, fixture.records().to_vec(), ep)?;
        let pin = digest(&parent)?;
        let m = Model::new(parent, &pin)?;
        let all: Vec<_> = (0..47).collect();
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
        for r in m.records() {
            assert_eq!(m.logits_cell(r.cell)?, m.parent.logits_cell(r.cell)?);
        }
    }
    Ok(())
}

#[test]
fn saved_gates_anchor_rows_work_and_external_parent_are_reconstructed() -> Result<()> {
    let m = make(Mode::Mini)?;
    let value = serde_json::to_value(&m)?;
    assert_eq!(m, Model::load(value.clone(), m.binding())?);
    for pointer in [
        "/gates/0",
        "/profiles/0/counts/0/1",
        "/anchor/0",
        "/radius",
        "/selection/weights/0/0",
        "/selection/canonical_ce",
        "/selection/work/score_records",
        "/selection/steps/0/count",
        "/bounds/buffer_bytes",
        "/binding/parent_sha256",
        "/parent/parent/context/source/coefficients/0",
    ] {
        let mut bad = value.clone();
        *bad.pointer_mut(pointer).unwrap() = json!(999);
        assert!(Model::load(bad, m.binding()).is_err(), "{pointer}");
    }
    let cell = m.rows.iter().position(Option::is_some).unwrap();
    let mut bad = value;
    bad["rows"][cell][0]["mass"] = json!(0.);
    assert!(Model::load(bad, m.binding()).is_err());
    assert!(Model::new(m.parent.clone(), "wrong").is_err());
    let mut rebound = serde_json::to_value(&m)?;
    rebound["parent"]["selection"]["canonical_ce"] = json!(0.);
    let mut binding = m.binding.clone();
    binding.parent_sha256 = digest(&rebound["parent"])?;
    rebound["binding"] = serde_json::to_value(&binding)?;
    assert!(Model::load(rebound, &binding).is_err());
    Ok(())
}

#[test]
fn anchor_identity_domains_limits_and_protocol_are_explicit() -> Result<()> {
    let mut m = make(Mode::Forward)?;
    for row in &mut m.selection.weights {
        row.clone_from(&m.anchor);
    }
    m.rows = model::rows(&m.parent, &m.selection.weights, &m.gates)?;
    for cell in 0..49 {
        assert_eq!(m.logits_cell(cell)?, m.parent.logits_cell(cell)?);
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
    let p: serde_json::Value = serde_json::from_str(PROTOCOL)?;
    assert_eq!(p["radius"], json!(RADIUS));
    assert_eq!(p["gate_count"], json!(GATE_COUNT));
    assert_eq!(p["line_search"], json!(LINE_SEARCH));
    Ok(())
}

#[test]
fn multiple_actual_complex_blocks_use_frozen_gates_and_canonical_products() -> Result<()> {
    let observations: Vec<_> = (0..64)
        .filter(|x| ![9, 27, 45, 53].contains(x))
        .map(|x| {
            let alias = usize::from((x % 8 + x / 8 < 7) ^ [4, 16, 30].contains(&x));
            [x, (x % 8 + x / 8) % 2 + 2 * alias]
        })
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
    assert!(source.blocks.len() >= 2);
    let records = observations
        .iter()
        .enumerate()
        .map(|(id, &[cell, label])| records::Record { id, cell, label })
        .collect();
    let parent = wrap(source, records, Mode::Forward)?;
    let profiles = gates::profiles(parent.source());
    let gates = gates::build(parent.source(), &profiles);
    let cache = diagnostic::Cache::new(&parent, &gates)?;
    let pin = digest(&parent)?;
    let mut m = Model::new(parent, &pin)?;
    let mut active = std::collections::BTreeSet::new();
    let mut shared = std::collections::BTreeMap::<usize, std::collections::BTreeSet<usize>>::new();
    for (position, row) in cache.rows.iter().enumerate() {
        let cell = m.records()[position].cell;
        let (group, _) = m.source().location(cell);
        active.insert(group);
        assert_eq!(row.gate, m.gates()[cell]);
        assert_eq!(row.gate, gates[cell]);
        shared.entry(row.gate).or_default().insert(group);
        let b = &m.source().blocks[group];
        assert!(bank::ambiguous(b));
        for j in [0, 17, 90] {
            let mut actual = vec![0.; m.source().vocabulary];
            for w in bank::weights(&m.bank()[j], &bank::distances(m.source(), cell)) {
                for (a, z) in actual
                    .iter_mut()
                    .zip(m.source().logits_state(group, w.state))
                {
                    *a += w.mass * z;
                }
            }
            for (c, &z) in actual.iter().enumerate() {
                let expected = row
                    .labels
                    .binary_search(&c)
                    .map_or(0., |i| row.logits[j][i]);
                assert!((z - expected).abs() < 1e-10);
            }
        }
    }
    assert!(active.len() >= 2);
    assert!(shared.values().any(|blocks| blocks.len() >= 2));
    for (g, local) in m.selection.weights.iter_mut().enumerate() {
        local.fill(0.);
        local[if g % 2 == 0 { 1 } else { 90 }] = 1.;
    }
    m.rows = model::rows(&m.parent, &m.selection.weights, &m.gates)?;
    let mut changed_groups = std::collections::BTreeSet::new();
    for cell in 0..64 {
        if m.rows[cell].is_some() {
            changed_groups.insert(m.source().location(cell).0);
            let (group, phi) = m.features(cell)?;
            let local = &m.selection.weights[m.gates[cell]];
            let effective = model::effective(&m.anchor, local);
            let sorted = bank::distances(m.source(), cell);
            let b = &m.source().blocks[group];
            let mut expected = vec![C::new(0., 0.); b.states()];
            for (context, &alpha) in m.bank().iter().zip(&effective) {
                for w in bank::weights(context, &sorted) {
                    for (f, p) in expected.iter_mut().enumerate() {
                        *p += alpha * w.mass * b.code[w.state * b.states() + f];
                    }
                }
            }
            for (a, b) in phi.iter().zip(expected) {
                assert!((*a - b).norm() < 1e-12);
            }
        }
    }
    assert!(changed_groups.len() >= 2);
    eprintln!(
        "actual nonzero Fourier fixture: {} source blocks, {} used shared gates, {} changed blocks",
        active.len(),
        shared.len(),
        changed_groups.len()
    );
    Ok(())
}

#[test]
fn count_boundaries_exclusion_and_label_renaming_are_exact() {
    for (counts, full, without_majority, without_minority) in [
        (vec![(2, 7), (9, 1)], 1, 2, 0),
        (vec![(2, 6), (9, 2)], 2, 3, 2),
        (vec![(2, 5), (9, 3)], 3, 4, 3),
        (vec![(2, 4), (9, 4)], 4, 4, 4),
    ] {
        let profile = Profile {
            counts: counts.clone(),
        };
        assert_eq!(gates::bucket(&profile, None), full);
        assert_eq!(gates::bucket(&profile, Some(2)), without_majority);
        assert_eq!(gates::bucket(&profile, Some(9)), without_minority);
        let renamed = Profile {
            counts: counts
                .into_iter()
                .rev()
                .map(|(label, n)| (100 - label, n))
                .collect(),
        };
        assert_eq!(gates::bucket(&renamed, None), full);
        assert_eq!(gates::bucket(&renamed, Some(98)), without_majority);
        assert_eq!(gates::bucket(&renamed, Some(91)), without_minority);
    }
    assert_eq!(gates::bucket(&Profile { counts: vec![] }, None), 0);
    assert_eq!(
        gates::bucket(
            &Profile {
                counts: vec![(10, 1)]
            },
            Some(10)
        ),
        0
    );
    assert_eq!(
        gates::bucket(
            &Profile {
                counts: vec![(10, 8)]
            },
            None
        ),
        0
    );
    assert_eq!(
        gates::bucket(
            &Profile {
                counts: vec![(1, 3), (4, 3), (9, 2)]
            },
            None
        ),
        4
    );
}

#[test]
fn repeated_records_cannot_restore_the_excluded_source_state() -> Result<()> {
    let observations: Vec<_> = (0..8)
        .map(|cell| [cell, if cell == 4 { 2 } else { 0 }])
        .collect();
    let report = kernel::discovery::run(kernel::discovery::Request {
        prime: 2,
        input: kernel::discovery::Input {
            n: 3,
            vocabulary: 5,
            observations: observations.clone(),
            policy: kernel::discovery::Policy::default(),
        },
    })?;
    let source = kernel::initialize(&report)?.model;
    assert_eq!(source.location(0).0, source.location(4).0);
    assert_eq!(source.location(0).0, source.location(8).0);
    let mut records: Vec<_> = observations
        .iter()
        .enumerate()
        .map(|(id, &[cell, label])| records::Record { id, cell, label })
        .collect();
    records.push(records::Record {
        id: 8,
        cell: 4,
        label: 2,
    });
    let parent = wrap(source, records, Mode::Forward)?;
    let pin = digest(&parent)?;
    let m = Model::new(parent, &pin)?;
    assert_eq!(m.gates[4], 0); // Removing the unique minority state leaves a pure profile.
    assert_eq!(m.gates[0], 2); // Removing a majority state gives minority fraction1/7.
    assert_eq!(m.gates[8], 1); // The unseen query keeps all8 unique source states.
    let cache = diagnostic::Cache::new(m.parent(), m.gates())?;
    assert_eq!(cache.rows[4].gate, cache.rows[8].gate);
    assert_eq!(cache.rows[4].gate, 0);
    assert_eq!(
        m.profiles[m.source().location(4).0].counts,
        vec![(0, 7), (2, 1)]
    );
    assert_eq!(m.selection.anchors[0].records, 2);
    for cell in 0..9 {
        let (block, state) = m.source().location(cell);
        let mut labels: Vec<_> = m.source().blocks[block]
            .observed_labels
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != state)
            .map(|(_, label)| *label)
            .collect();
        labels.sort_unstable();
        let mut runs = Vec::new();
        for label in labels {
            if let Some((old, count)) = runs.last_mut()
                && *old == label
            {
                *count += 1;
            } else {
                runs.push((label, 1usize));
            }
        }
        let total: usize = runs.iter().map(|(_, n)| n).sum();
        let minority = total - runs.iter().map(|(_, n)| *n).max().unwrap_or(0);
        let ratio = if total == 0 {
            0.
        } else {
            minority as f64 / total as f64
        };
        let expected = if ratio == 0. {
            0
        } else if ratio <= 0.125 {
            1
        } else if ratio <= 0.25 {
            2
        } else if ratio <= 0.375 {
            3
        } else {
            4
        };
        assert_eq!(m.gates[cell], expected);
    }
    Ok(())
}

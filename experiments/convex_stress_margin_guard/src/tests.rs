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
    let mixture = parent::parent::Model::new(scaled, ep, &pin)?;
    let pin = digest(&mixture)?;
    parent::Model::new(mixture, &pin)
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
fn all_modes_keep_raw_margins_while_changing_actual_confidence() -> Result<()> {
    let mut changed = 0;
    for mode in [Mode::Full, Mode::Mini, Mode::Forward] {
        let m = make(mode)?;
        let s = m.selection();
        assert!(s.canonical_ce <= s.initial_ce + 1e-12);
        assert!(s.initial_ce <= m.parent.selection().canonical_ce + 1e-12);
        assert_eq!(m.bounds.bank_size, 93);
        assert_eq!(s.weights.len(), GATE_COUNT);
        assert_eq!(s.anchors.iter().map(|c| c.records).sum::<usize>(), 47);
        let expected = match mode {
            Mode::Full => 64,
            Mode::Mini => 48,
            Mode::Forward => 0,
        };
        assert_eq!(s.steps.len(), expected);
        assert_eq!(s.work.gradient_batches, expected);
        assert_eq!(
            s.work.gradient_records,
            match mode {
                Mode::Full => 64 * 47,
                Mode::Mini => 8 * 47,
                Mode::Forward => 0,
            }
        );
        assert_eq!(
            s.work.optimizer_updates,
            s.steps.iter().filter(|s| s.changed_rows > 0).count()
        );
        assert_eq!(
            s.work.local_row_updates,
            s.steps.iter().map(|s| s.changed_rows).sum::<usize>()
        );
        assert_eq!(m.bounds.guarded_cells, 3);
        assert!(m.certificate.raw_decisions_unchanged);
        assert!(s.work.guard_query_visits as u64 <= m.bounds.max_guard_query_visits);
        assert_eq!(s.work.guard_class_visits, s.work.guard_query_visits * 7);
        assert!(m.bounds.buffer_bytes < 56 * 1024 * 1024 * 1024);
        assert!(m.bounds.total_scalar_work < WORK_CAP);
        assert_eq!(
            m.bounds.inherited_scalar_work,
            m.parent.bounds().total_scalar_work
        );
        if mode == Mode::Mini {
            assert_eq!(s.steps.iter().filter(|s| s.count == 7).count(), 8);
        }
        if mode == Mode::Forward {
            assert_eq!(s.work.optimizer_updates, 0);
        }
        for row in &s.weights {
            assert!(row.iter().all(|a| *a >= 0.));
            assert!((row.iter().sum::<f64>() - 1.).abs() < 1e-12);
        }
        for step in &s.steps {
            assert!(step.after_ce <= step.before_ce);
            assert!(step.caps.iter().all(|c| (0.0..=1.0).contains(c)));
            assert!(step.gamma <= step.proposed_gamma);
        }
        for cell in 0..49 {
            let old = m.parent.logits_cell(cell)?;
            let new = m.logits_cell(cell)?;
            assert!(margins_hold(&old, &new), "mode={mode:?},cell={cell}");
            changed += usize::from(old != new);
        }
        eprintln!(
            "{mode:?}: diagnostic {:.9}->{:.9}, guarded={}, raw flips=0, initial changed rows={}, updates={}, rejected={}",
            m.parent.selection().canonical_ce,
            s.canonical_ce,
            m.bounds.guarded_cells,
            s.initial_step.changed_rows,
            s.work.optimizer_updates,
            s.work.rejected_proposals
        );
    }
    assert!(
        changed > 0,
        "the guard must permit a genuine confidence change"
    );
    Ok(())
}

#[test]
fn diagnostic_matches_dense_ce_slopes_and_original_short_tail() -> Result<()> {
    let parent = fixture(Mode::Full)?;
    let profiles = gates::profiles(parent.source());
    let gates = gates::build(parent.source(), &profiles);
    let cache = diagnostic::Cache::new(&parent, &gates)?;
    let weights = vec![vec![1. / 93.; 93]; GATE_COUNT];
    let actual = cache.evaluate(&weights, 0, 47, true);
    let mut dense = 0.;
    for row in &cache.rows {
        let sparse = row.mixture(&weights[row.gate]);
        let mut z = vec![0.; cache.vocabulary];
        for (&label, &value) in row.labels.iter().zip(&sparse) {
            z[label] = value;
        }
        dense += kernel::training::ce(&z, row.labels[row.target]);
        assert_eq!(row.logits[0], row.anchor);
        assert_eq!(
            row.logits[1],
            row.anchor.iter().map(|v| v * 0.5).collect::<Vec<_>>()
        );
    }
    assert!((actual.loss - dense / 47.).abs() < 1e-12);
    for g in 0..GATE_COUNT {
        for j in [0, 1, 2, 19, 92] {
            let (mut plus, mut minus) = (weights.clone(), weights.clone());
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
    for g in 0..GATE_COUNT {
        for j in 0..93 {
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
fn complete_complex_products_bounds_and_cached_caps_are_checked() -> Result<()> {
    let m = make(Mode::Forward)?;
    assert!(
        m.source()
            .blocks
            .iter()
            .flat_map(|b| &b.code)
            .any(|z| z.im.abs() > 0.1)
    );
    let guard = guard::Guard::new(&m.parent, &m.gates)?;
    let mut work = Work::default();
    let caps = guard.initial_caps(93, &mut work);
    assert_eq!(work.cap_class_visits, 3 * 93 * 7);
    let mut rejected = 0;
    let current: Vec<_> = guard.queries.iter().map(|q| q.reference.clone()).collect();
    for j in 1..93 {
        let mut weights = vec![anchor(93); GATE_COUNT];
        for row in &mut weights {
            selection::update_row(row, j, 1.);
        }
        rejected += usize::from(
            guard
                .verify(&m.parent, &weights, &current, &[true; 5], &mut work)
                .is_none(),
        );
    }
    assert!(
        rejected > 0,
        "adversarial bank vertices must exercise rejection"
    );
    assert!(caps.iter().flatten().any(|x| *x < 1.));
    for cell in 0..49 {
        let (block, phi) = m.features(cell)?;
        assert!(
            phi.iter()
                .all(|z| z.re.abs() <= 1. + 1e-12 && z.im.abs() <= 1. + 1e-12)
        );
        let b = &m.source().blocks[block];
        let v = m.source().vocabulary;
        let mut dense = vec![0.; v];
        for (f, p) in phi.iter().enumerate() {
            let h = 2 * (b.offset + f) * v;
            for (c, z) in dense.iter_mut().enumerate() {
                *z += p.re * m.source().coefficients[h + c]
                    - p.im * m.source().coefficients[h + v + c];
            }
        }
        for (a, b) in dense.iter().zip(m.logits_cell(cell)?) {
            assert!((a - b).abs() < 1e-10);
            assert!(a.abs() <= m.source().radius + 1e-10);
        }
    }
    Ok(())
}

#[test]
fn exact_near_and_signed_zero_ties_reject_even_tiny_decision_flips() {
    assert!(margins_hold(&[1., 1., 0.], &[0.5, 0.5, 0.]));
    assert!(!margins_hold(&[1., 1., 0.], &[0.5, 0.5 + 1e-15, 0.]));
    assert!(margins_hold(&[0., 2e-15], &[0., 1e-15]));
    assert!(!margins_hold(&[0., 2e-15], &[0., 0.]));
    assert_eq!(argmax(&[-0., 0.]), 1);
    assert!(margins_hold(&[-0., 0.], &[-0., 0.]));
    assert!(!margins_hold(&[-0., 0.], &[0., 0.]));
    assert!(margins_hold(&[-1., -0.], &[-0.5, -0.]));
    assert!(!margins_hold(&[0., 1.], &[0., f64::NAN]));
    assert!(!margins_hold(&[], &[]));
    assert!(!margins_hold(&[0., 1.], &[1.]));
    assert_eq!(guard::cap(&[2., 0.], &[2., 0.], &[0., 2.], 0), 0.25);
    assert_eq!(guard::cap(&[1., 1.], &[1., 1.], &[0., 2.], 0), 0.);
}

#[test]
fn every_actual_training_mode_preserves_original_head_derivatives() -> Result<()> {
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
fn sealed_constraints_coefficients_costs_and_rehashed_parent_are_replayed() -> Result<()> {
    let m = make(Mode::Mini)?;
    let value = serde_json::to_value(&m)?;
    assert_eq!(m, Model::load(value.clone(), m.binding())?);
    for pointer in [
        "/gates/0",
        "/profiles/0/counts/0/1",
        "/radius",
        "/retention",
        "/selection/weights/0/0",
        "/selection/canonical_ce",
        "/selection/work/guard_class_visits",
        "/selection/steps/0/count",
        "/selection/candidates/0/1/cap",
        "/certificate/guarded_cells",
        "/certificate/final_scores_sha256",
        "/bounds/buffer_bytes",
        "/binding/parent_sha256",
        "/parent/parent/parent/context/source/coefficients/0",
    ] {
        let mut bad = value.clone();
        *bad.pointer_mut(pointer).unwrap() = json!(999);
        assert!(Model::load(bad, m.binding()).is_err(), "{pointer}");
    }
    if let Some(cell) = m.rows.iter().position(Option::is_some) {
        let mut bad = value.clone();
        bad["rows"][cell]["reference_mass"] = json!(0.);
        assert!(Model::load(bad, m.binding()).is_err());
    }
    assert!(Model::new(m.parent.clone(), "wrong").is_err());
    let mut rebound = value;
    rebound["parent"]["selection"]["canonical_ce"] = json!(0.);
    let mut binding = m.binding.clone();
    binding.parent_sha256 = digest(&rebound["parent"])?;
    rebound["binding"] = serde_json::to_value(&binding)?;
    assert!(Model::load(rebound, &binding).is_err());
    Ok(())
}

#[test]
fn anchor_identity_and_domains_are_exact() -> Result<()> {
    let mut m = make(Mode::Forward)?;
    m.selection.weights = vec![anchor(93); 5];
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
    assert_eq!(p["margin_retention"], json!(RETENTION));
    assert_eq!(p["scalar_work_limit"], json!(WORK_CAP));
    assert_eq!(p["backtracks"], json!(BACKTRACKS));
    let limit = 56u128 * 1024 * 1024 * 1024;
    assert!(binding::enforce_limits(limit, WORK_CAP as u128).is_ok());
    assert!(binding::enforce_limits(limit + 1, 0).is_err());
    assert!(binding::enforce_limits(0, WORK_CAP as u128 + 1).is_err());
    assert!(binding::enforce_limits(u128::MAX, u128::MAX).is_err());
    Ok(())
}

#[test]
fn multiple_nonzero_blocks_share_gates_without_sharing_source_coefficients() -> Result<()> {
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
    let records = observations
        .iter()
        .enumerate()
        .map(|(id, &[cell, label])| records::Record { id, cell, label })
        .collect();
    let parent = wrap(source, records, Mode::Forward)?;
    let pin = digest(&parent)?;
    let m = Model::new(parent, &pin)?;
    let guard = guard::Guard::new(&m.parent, &m.gates)?;
    let cache = diagnostic::Cache::new(&m.parent, &m.gates)?;
    let mut shared = std::collections::BTreeMap::<usize, std::collections::BTreeSet<usize>>::new();
    for q in &guard.queries {
        let block = m.source().location(q.cell).0;
        shared.entry(q.gate).or_default().insert(block);
        assert_eq!(q.reference, m.parent.logits_cell(q.cell)?);
        for j in [0, 17, 90] {
            let mut coefficients = vec![0.; 93];
            coefficients[j + 2] = 1.;
            let row = model::row(m.source(), q.cell, &coefficients, &q.maps);
            let actual = model::logits(m.source(), q.cell, &row, &q.reference);
            for (c, z) in actual.iter().enumerate() {
                let affine = (1. - RADIUS) * q.reference[c] + RADIUS * q.logits[j + 2][c];
                assert!((z - affine).abs() < 1e-10);
            }
        }
    }
    assert!(shared.values().any(|blocks| blocks.len() >= 2));
    for (position, row) in cache.rows.iter().enumerate() {
        let cell = m.records()[position].cell;
        let block = m.source().location(cell).0;
        let sorted = bank::distances(m.source(), cell);
        assert_eq!(row.gate, m.gates[cell]);
        for j in [0, 17, 90] {
            let mut actual = [0.; 8];
            for w in bank::weights(&m.bank()[j], &sorted) {
                for (a, z) in actual
                    .iter_mut()
                    .zip(m.source().logits_state(block, w.state))
                {
                    *a += w.mass * z;
                }
            }
            for (c, &z) in actual.iter().enumerate() {
                let expected = row
                    .labels
                    .binary_search(&c)
                    .map_or(0., |i| row.logits[j + 2][i]);
                assert!((z - expected).abs() < 1e-10);
            }
        }
    }
    for cell in 0..64 {
        assert!(margins_hold(
            &m.parent.logits_cell(cell)?,
            &m.logits_cell(cell)?
        ));
    }
    eprintln!(
        "multiple-block guard: {} query cells, {} shared gates",
        guard.queries.len(),
        shared.len()
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

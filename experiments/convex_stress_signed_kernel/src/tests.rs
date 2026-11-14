use crate::*;
use convex_stress_shared_roles::training;
use serde_json::json;

fn crossing_fixture() -> Result<(kernel::Model, Vec<records::Record>)> {
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
    let records = (0..45)
        .map(|id| {
            let [cell, label] = observations[id % observations.len()];
            records::Record { id, cell, label }
        })
        .collect();
    Ok((source, records))
}

pub(crate) fn fixture() -> Result<(kernel::Model, Vec<records::Record>)> {
    // Three adjacent diagonals: opposite checkerboard corners force a raw
    // alias modulo3. Overlapping class hulls leave room for a signed map.
    // The constructor receives observations only, with no unseen target table.
    let observations: Vec<_> = (0usize..49)
        .filter(|cell| (cell % 7).abs_diff(cell / 7) <= 1)
        .map(|cell| [cell, if cell % 7 == cell / 7 { 2 } else { 0 }])
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
    let records = (0..45)
        .map(|id| {
            let [cell, label] = observations[id % observations.len()];
            records::Record { id, cell, label }
        })
        .collect();
    Ok((source, records))
}

pub(crate) fn wrap(
    source: kernel::Model,
    records: Vec<records::Record>,
    mode: Mode,
    updates: usize,
) -> Result<parent::Model> {
    let ep = Endpoint {
        mode,
        source_reference: "signed-kernel-synthetic".into(),
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
    let r = regions::Model::new(g, &pin)?;
    let pin = digest(&r)?;
    let exclusion = parent::parent::Model::new(r, &pin)?;
    let pin = digest(&exclusion)?;
    parent::Model::new(exclusion, &pin)
}

pub(crate) fn make() -> Result<Model> {
    let (source, records) = fixture()?;
    let parent = wrap(source, records, Mode::Forward, 0)?;
    let pin = digest(&parent)?;
    Model::new(parent, &pin)
}

fn independent_products(m: &Model) -> Result<usize> {
    let mut checked = 0;
    for cell in 0..m.source().n.pow(2) {
        let Some(row) = &m.rows[cell] else {
            assert_eq!(m.features(cell)?, m.parent.features(cell)?);
            assert_eq!(m.logits_cell(cell)?, m.parent.logits_cell(cell)?);
            continue;
        };
        checked += 1;
        assert!(model::eligible(m.parent(), cell));
        let mut expected: Vec<_> = m
            .parent
            .logits_cell(cell)?
            .into_iter()
            .map(|z| row.parent_mass * z)
            .collect();
        for (state, &mass) in row.weights.iter().enumerate() {
            for (z, value) in expected
                .iter_mut()
                .zip(m.source().logits_state(row.block, state))
            {
                *z += mass * value;
            }
        }
        let actual = m.logits_cell(cell)?;
        for (a, b) in actual.iter().zip(expected) {
            assert!((a - b).abs() < 1e-10);
        }
        let mut weights = vec![row.parent_mass];
        weights.extend(&row.weights);
        assert!(solver::upper_mass(&weights)? <= 1.);
        assert!(
            m.features(cell)?
                .1
                .iter()
                .all(|p| p.re.abs() <= 1. + 1e-12 && p.im.abs() <= 1. + 1e-12)
        );
        assert!(actual.iter().all(|z| z.abs() <= m.source().radius + 1e-10));
    }
    Ok(checked)
}

#[test]
fn all_actual_modes_keep_original_products_records_and_tail_gradients() -> Result<()> {
    let (source, records) = fixture()?;
    for policy in training::policies(records.len(), 8)? {
        let mode = match policy.mode {
            training::Mode::Full => Mode::Full,
            training::Mode::Mini => Mode::Mini,
            training::Mode::Forward => Mode::Forward,
        };
        let out = training::run(&source, &records, policy)?;
        training::replay(&source, &records, &out)?;
        let old_steps = out.steps.len();
        assert_eq!(
            old_steps,
            match mode {
                Mode::Full => 1,
                Mode::Mini => 6,
                Mode::Forward => 0,
            }
        );
        let parent = wrap(out.model, records.clone(), mode, old_steps)?;
        let pin = digest(&parent)?;
        let m = Model::new(parent, &pin)?;
        assert_eq!(m.records(), records);
        assert_eq!(m.source().coefficients, m.parent.source().coefficients);
        assert_eq!(m.endpoint().source_optimizer_updates, Some(old_steps));
        assert_eq!(
            m.work.gradient_batches,
            match mode {
                Mode::Full => 64,
                Mode::Mini => 48,
                Mode::Forward => 0,
            }
        );
        assert_eq!(m.selection.steps.len(), m.work.gradient_batches);
        assert!(
            m.selection
                .steps
                .iter()
                .all(|s| s.after_ce <= s.before_ce + 1e-10)
        );
        if mode == Mode::Mini {
            assert_eq!(m.selection.steps.last().unwrap().count, 5);
        }
        if mode == Mode::Forward {
            assert_eq!(m.work.optimizer_updates, 0);
        }
        assert!(m.statistics.eligible_cells > 0);
        assert!(independent_products(&m)? > 0);
        let all: Vec<_> = (0..records.len()).collect();
        for batch in [&all[..], &all[..8], &all[40..]] {
            let old = m.parent.batch_gradient(batch)?;
            let actual = m.batch_gradient(batch)?;
            assert_eq!(old.loss, actual.loss);
            assert_eq!(old.coefficients, actual.coefficients);
            let direct = batch
                .iter()
                .map(|&i| {
                    kernel::training::ce(&m.logits_cell(records[i].cell).unwrap(), records[i].label)
                })
                .sum::<f64>()
                / batch.len() as f64;
            assert_eq!(direct, actual.loss);
        }
        eprintln!(
            "{mode:?}: source_updates={old_steps} new_gradient_batches={} rows={} negative_rows={} baseline_ce={} selected_ce={}",
            m.work.gradient_batches,
            m.statistics.changed_feature_rows,
            m.statistics.negative_weight_rows,
            m.selection.candidates[0].canonical_ce,
            m.selection.canonical_ce
        );
    }
    Ok(())
}

#[test]
fn saved_fits_selection_rows_costs_and_source_mutations_are_rejected() -> Result<()> {
    let m = make()?;
    let value = serde_json::to_value(&m)?;
    assert_eq!(m, Model::load(value.clone(), m.binding())?);
    let cell = m.rows.iter().position(Option::is_some).unwrap();
    for mutation in 0..10 {
        let mut bad = value.clone();
        match mutation {
            0 => bad["bank"][1]["ridge"] = json!(2.),
            1 => bad["fits"][0]["inverse_sha256"] = json!("changed"),
            2 => bad["selection"]["weights"][0] = json!(0.75),
            3 => bad["rows"][cell]["weights"][0] = json!(99.),
            4 => bad["rows"][cell]["parent_mass"] = json!(-1.),
            5 => bad["diagnostic_sha256"] = json!("changed"),
            6 => bad["work"]["kernel_matrices"] = json!(0),
            7 => bad["bounds"]["construction_scalar_work"] = json!(0),
            8 => bad["statistics"]["changed_feature_rows"] = json!(0),
            _ => bad["parent"]["binding"]["parent_sha256"] = json!("changed"),
        }
        assert!(Model::load(bad, m.binding()).is_err(), "mutation{mutation}");
    }
    assert!(Model::new(m.parent.clone(), "wrong").is_err());
    let mut wrong = m.binding.clone();
    wrong.protocol_sha256 = "wrong".into();
    assert!(Model::load(value, &wrong).is_err());
    Ok(())
}

#[test]
fn unchanged_masks_bounds_domains_and_heldout_targets_are_explicit() -> Result<()> {
    let m = make()?;
    assert_eq!(m.bounds, forecast(m.parent())?);
    assert_eq!(
        m.bounds.total_scalar_work,
        m.bounds.inherited_scalar_work
            + m.bounds.construction_scalar_work
            + m.bounds.optimization_scalar_work
    );
    assert!(binding::estimate(m.parent(), BUFFER_LIMIT, 0).is_err());
    assert!(binding::estimate(m.parent(), 0, WORK_LIMIT).is_err());
    for prefix in [vec![], vec![1], vec![0, -1], vec![0, 7], vec![0; 4097]] {
        assert!(m.logits(&prefix).is_err());
    }
    assert!(m.logits(&vec![0; 4096]).is_ok());
    assert!(m.features(49).is_err());
    assert!(m.logits_cell(49).is_err());
    for batch in [vec![], vec![0, 0], vec![45]] {
        assert!(m.batch_gradient(&batch).is_err());
    }
    for record in m.records() {
        assert!(m.rows[record.cell].is_none());
    }
    let (source, records) = crossing_fixture()?;
    let crossed = wrap(source, records, Mode::Forward, 0)?;
    assert!(crossed.rows()[21].is_some());
    let pin = digest(&crossed)?;
    let retained = Model::new(crossed, &pin)?;
    assert!(retained.rows[21].is_none());
    assert_eq!(retained.logits_cell(21)?, retained.parent.logits_cell(21)?);
    let pin = digest(m.parent())?;
    let other = Model::new(m.parent.clone(), &pin)?;
    assert_eq!(m, other); // Neither construction receives either held-out target table.
    let cell = m.rows.iter().position(Option::is_some).unwrap();
    let z = m.logits_cell(cell)?;
    assert_ne!(kernel::training::ce(&z, 0), kernel::training::ce(&z, 2));
    assert_eq!(
        m.statistics.retained_parent_cells + m.statistics.eligible_cells,
        49
    );
    assert_eq!(
        m.statistics.changed_feature_rows + m.statistics.retained_baseline_rows,
        m.statistics.eligible_cells
    );
    Ok(())
}

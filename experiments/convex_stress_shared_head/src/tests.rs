use crate::*;
use training::{Mode, Policy};

fn fixture() -> Input {
    let cells = [7, 8, 14, 15, 11, 12, 18, 19];
    Input {
        parent_sha256: "1".repeat(64),
        source_sha256: "2".repeat(64),
        n: 7,
        vocabulary: 7,
        blocks: 1,
        route: vec![0; 49],
        records: (0..37)
            .map(|id| {
                let cell = cells[id % cells.len()];
                Record {
                    id,
                    cell,
                    label: if cell % 7 < 3 { 2 } else { 5 },
                }
            })
            .collect(),
    }
}
fn argmax(z: &[f64]) -> usize {
    z.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
        .unwrap()
        .0
}

#[test]
fn complete_complex_products_match_independent_polynomial_after_arbitrary_head_change() -> Result<()>
{
    let mut m = Model::zero(fixture())?;
    for (i, a) in m.coefficients.iter_mut().enumerate() {
        *a = ((i * 17 % 31) as f64 - 15.) / 7.;
    }
    m.validate()?;
    for cell in 0..49 {
        let phi = features::encode(7, cell);
        assert!(
            phi.iter()
                .all(|p| p.re.abs() <= 1. + 1e-12 && p.im.abs() <= 1. + 1e-12)
        );
        let (w, _) = features::row(7, cell);
        assert!(w.iter().map(|a| a.abs()).sum::<f64>() <= 1. + 1e-12);
        for (c, z) in m.logits_cell(cell)?.iter().enumerate() {
            let p = features::polynomial(&m, 0, c);
            let independent: f64 = w.iter().zip(p).map(|(a, b)| a * b).sum();
            assert!((z - independent).abs() < 1e-11);
        }
    }
    Ok(())
}

#[test]
fn initialization_learns_nonidentity_labels_and_uses_same_features_on_new_inputs() -> Result<()> {
    let input = fixture();
    let (m, report) = initialize::run(input.clone())?;
    assert_eq!(report.blocks[0].method, "separated_intervals");
    assert_eq!(report.searched_directions, 16);
    assert_eq!(report.direction_records, 16 * 37);
    for r in &input.records {
        assert_eq!(argmax(&m.logits_cell(r.cell)?), r.label);
    }
    for x in 0..7 {
        for (y, label) in [(0, 2), (6, 5)] {
            assert_eq!(argmax(&m.logits_cell(x * 7 + y)?), label);
        }
    }
    let mut relabel = input;
    for r in &mut relabel.records {
        r.label = if r.label == 2 { 6 } else { 1 };
    }
    let (permuted, _) = initialize::run(relabel.clone())?;
    for r in &relabel.records {
        assert_eq!(argmax(&permuted.logits_cell(r.cell)?), r.label);
    }
    Ok(())
}

#[test]
fn overlapping_linear_intervals_use_explicit_quadratic_moments() -> Result<()> {
    let mut input = fixture();
    input.records = [(0, 2), (48, 2), (6, 5), (42, 5)]
        .into_iter()
        .enumerate()
        .map(|(id, (cell, label))| Record { id, cell, label })
        .collect();
    let (m, r) = initialize::run(input.clone())?;
    assert_eq!(r.blocks[0].method, "quadratic_moments");
    assert_eq!(r.blocks[0].selected, None);
    assert!(r.blocks[0].candidates.iter().all(|c| c.gap <= 0.));
    assert_eq!(r.moment_records, 4);
    for r in &input.records {
        assert_eq!(argmax(&m.logits_cell(r.cell)?), r.label);
    }
    Ok(())
}

#[test]
fn full_head_derivative_matches_finite_differences_and_weighted_real_tails() -> Result<()> {
    let mut m = Model::zero(fixture())?;
    for (i, x) in m.coefficients.iter_mut().enumerate() {
        *x = ((i * 13 % 23) as f64 - 11.) / 40.;
    }
    let cache = training::encode(&m.input)?;
    let all: Vec<_> = (0..37).collect();
    let (loss, g) = training::gradient(&m, &cache, &all)?;
    assert!((loss - training::objective(&m, &cache, &all)?).abs() < 1e-12);
    assert!(g.iter().any(|a| a.abs() > 1e-3));
    let h = 1e-5;
    for (i, &derivative) in g.iter().enumerate() {
        let old = m.coefficients[i];
        m.coefficients[i] = old + h;
        let plus = training::objective(&m, &cache, &all)?;
        m.coefficients[i] = old - h;
        let minus = training::objective(&m, &cache, &all)?;
        m.coefficients[i] = old;
        assert!(
            ((plus - minus) / (2. * h) - derivative).abs() < 1e-8,
            "coordinate{i}"
        );
    }
    let mut combined = vec![0.; g.len()];
    let mut combined_loss = 0.;
    for batch in all.chunks(8) {
        let (loss, local) = training::gradient(&m, &cache, batch)?;
        let weight = batch.len() as f64 / all.len() as f64;
        combined_loss += weight * loss;
        for (a, b) in combined.iter_mut().zip(local) {
            *a += weight * b;
        }
    }
    assert!((combined_loss - loss).abs() < 1e-12);
    assert!(combined.iter().zip(g).all(|(a, b)| (a - b).abs() < 1e-12));
    Ok(())
}

#[test]
fn projection_is_global_per_class_and_satisfies_soft_threshold_conditions() -> Result<()> {
    let mut input = fixture();
    input.blocks = 2;
    input.route[48] = 1;
    let mut m = Model::zero(input)?;
    let v = m.input.vocabulary;
    let positions = [3, WIDTH * v + 3, 2 * v + 3];
    for (&i, x) in positions.iter().zip([1.25, -1., 0.5]) {
        m.coefficients[i] = RADIUS * x;
    }
    let before = m.coefficients.clone();
    m.project();
    m.validate()?;
    let threshold = 0.625 * RADIUS;
    for (i, &x) in before.iter().enumerate() {
        let expected = if i % v == 3 {
            x.signum() * (x.abs() - threshold).max(0.)
        } else {
            x
        };
        assert!((m.coefficients[i] - expected).abs() < 1e-8);
    }
    let once = m.coefficients.clone();
    m.project();
    assert_eq!(m.coefficients, once);
    Ok(())
}

#[test]
fn all_modes_keep_actual_ce_records_real_tails_and_zero_forward_gradients() -> Result<()> {
    for mode in [Mode::Full, Mode::Mini, Mode::Forward] {
        let p = Policy {
            mode,
            iterations: if mode == Mode::Forward { 0 } else { 2 },
            batch: 8,
            learning_rate: 1.,
        };
        let input = fixture();
        let out = training::run(input.clone(), p)?;
        training::replay(&input, &out)?;
        if mode == Mode::Forward {
            assert!(out.steps.is_empty());
            assert_eq!(out.work.gradient_batches, 0);
            assert_eq!(out.work.updates, 0);
            assert_eq!(out.initial_ce, out.final_ce);
            assert_eq!(out.initialization.direction_records, 592);
        } else {
            assert!(out.work.effective_updates > 0);
            assert!(out.final_ce < out.initial_ce);
            assert!(out.steps.iter().any(|s| s.gradient_l2 > 0.));
            if mode == Mode::Mini {
                assert_eq!(out.steps.last().unwrap().record_ids, [32, 33, 34, 35, 36]);
            }
        }
        let mut changed = out.clone();
        changed.model.coefficients[0] += 0.01;
        assert!(training::replay(&input, &changed).is_err());
        changed = out.clone();
        changed.work.ce_examples += 1;
        assert!(training::replay(&input, &changed).is_err());
    }
    Ok(())
}

#[test]
fn dimensions_and_records_are_checked_before_construction_or_gradient_work() -> Result<()> {
    let good = fixture();
    let mut bad = good.clone();
    bad.records[1].id = bad.records[0].id;
    assert!(Model::zero(bad).is_err());
    bad = good.clone();
    bad.records[1].cell = bad.records[0].cell;
    bad.records[1].label = 6;
    assert!(Model::zero(bad).is_err());
    bad = good;
    bad.n = 257;
    bad.vocabulary = 257;
    bad.route = vec![0; 257 * 257];
    bad.records = (0..66_049)
        .map(|id| Record {
            id,
            cell: 0,
            label: 2,
        })
        .collect();
    assert!(limits::forecast(&bad, &training::policies()?[0]).is_err());
    let mut p = Policy::forward();
    p.iterations = 1;
    assert!(p.validate().is_err());
    Ok(())
}

#[test]
fn extraction_binds_complete_source_and_original_records_ignoring_unrelated_metrics() -> Result<()>
{
    use std::collections::BTreeMap;
    let fixture = fixture();
    let unique: BTreeMap<_, _> = fixture.records.iter().map(|r| (r.cell, r.label)).collect();
    let found = kernel::discovery::run(kernel::discovery::Request {
        prime: 3,
        input: kernel::discovery::Input {
            n: 7,
            vocabulary: 7,
            observations: unique.into_iter().map(|(x, y)| [x, y]).collect(),
            policy: kernel::discovery::Policy::default(),
        },
    })?;
    let source = kernel::initialize(&found)?.model;
    let mut parent = serde_json::json!({"prior":{"parent":{"context":{"source":source,"records":fixture.records}}},"metrics":{"score":1.}});
    let expected = Input::from_source(&source, fixture.records.clone(), "a".repeat(64))?;
    assert_eq!(
        inspection::extract(&parent, "prior", &"a".repeat(64))?,
        expected
    );
    parent["metrics"]["score"] = serde_json::json!(-100.);
    assert_eq!(
        inspection::extract(&parent, "prior", &"a".repeat(64))?,
        expected
    );
    assert!(inspection::extract(&parent, "supplement", &"a".repeat(64)).is_err());
    parent["prior"]["parent"]["context"]["records"][0]["label"] = serde_json::json!(0);
    assert!(inspection::extract(&parent, "prior", &"a".repeat(64)).is_err());
    Ok(())
}

#[test]
fn complete_original_shapes_are_admitted_before_fitting_without_reading_labels() -> Result<()> {
    let protocol: serde_json::Value = serde_json::from_str(PROTOCOL)?;
    assert_eq!(protocol["limits"]["kernel_work"], limits::WORK_LIMIT as u64);
    for blocks in [389, 97] {
        let input = Input {
            parent_sha256: "1".repeat(64),
            source_sha256: "2".repeat(64),
            n: 256,
            vocabulary: 256,
            blocks,
            route: vec![0; 256 * 256],
            records: (0..4704)
                .map(|id| Record {
                    id,
                    cell: 0,
                    label: 2,
                })
                .collect(),
        };
        for policy in training::policies()? {
            let bound = limits::forecast(&input, &policy)?;
            if blocks == 389 && policy.mode != Mode::Forward {
                assert!(bound.total_kernel_work > 256_000_000_000);
            }
            assert!(bound.total_kernel_work <= limits::WORK_LIMIT as u64);
            let mut relabel = input.clone();
            for r in &mut relabel.records {
                r.label = 191;
            }
            relabel.records.reverse();
            assert_eq!(limits::forecast(&relabel, &policy)?, bound);
            eprintln!(
                "blocks={blocks} mode={:?} work={} buffers={}",
                policy.mode, bound.total_kernel_work, bound.buffer_bytes
            );
        }
    }
    Ok(())
}

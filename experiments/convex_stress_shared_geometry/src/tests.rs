use crate::*;
use features::{Bank, Encoder};
fn fixture() -> Input {
    let n = 9;
    Input {
        parent_sha256: "1".repeat(64),
        source_sha256: "2".repeat(64),
        n,
        vocabulary: n,
        blocks: 3,
        route: (0..n * n).map(|cell| cell / n / 3).collect(),
        records: (0..85)
            .map(|id| {
                let cell = id % (n * n);
                let b = cell / n / 3;
                Record {
                    id,
                    cell,
                    label: if cell % n < 3 + b { b + 1 } else { b + 6 },
                }
            })
            .collect(),
    }
}
fn arbitrary(bank: Bank) -> Result<Model> {
    let input = fixture();
    let q = 2 * bank.states() + input.blocks;
    Ok(Model {
        input,
        protocol_sha256: protocol_hash()?,
        bank,
        decoder: vec![vec![1, 6], vec![2, 7], vec![3, 8]],
        theta: (0..q).map(|i| ((i * 7 % 11) as f64 - 5.) / 19.).collect(),
    })
}
fn argmax(z: &[f64]) -> usize {
    (0..z.len())
        .max_by(|&a, &b| z[a].total_cmp(&z[b]).then_with(|| b.cmp(&a)))
        .unwrap()
}

#[test]
fn actual_complete_complex_products_and_charged_budget() -> Result<()> {
    for bank in features::BANKS {
        let mut m = arbitrary(bank)?;
        m.validate()?;
        let (weights, offset) = m.weights();
        assert!(m.expanded_mass() <= offset + projection::mass(&m.theta, &weights) + 1e-9);
        for cell in 0..m.input.n.pow(2) {
            assert!(training::audit(&m, cell)? < 1e-10);
            let row = Encoder::new(bank).row(&m.input, cell, 0);
            assert!(row.phi.iter().all(|p| p.norm() <= 1. + 1e-12));
        }
        for x in &mut m.theta {
            *x *= 1e8;
        }
        assert!(m.validate().is_err());
        m.project()?;
        m.validate()?;
        assert!(m.expanded_mass() <= RADIUS * (1. + 1e-10));
    }
    Ok(())
}
#[test]
fn weighted_projection_matches_independent_kkt_solution() -> Result<()> {
    let original = [3., -4., 5., -2.];
    let weights = [1., 2., 3., 0.];
    let radius = 4.;
    let mut theta = original;
    let lambda = projection::project(&mut theta, &weights, radius)?;
    let (mut low, mut high) = (0., 10.);
    for _ in 0..100 {
        let mid = (low + high) / 2.;
        let mass = original
            .iter()
            .zip(weights)
            .map(|(x, w)| w * (x.abs() - mid * w).max(0.))
            .sum::<f64>();
        if mass > radius {
            low = mid;
        } else {
            high = mid;
        }
    }
    assert!((lambda - high).abs() < 1e-12);
    for ((&a, &b), w) in original.iter().zip(&theta).zip(weights) {
        assert!((b - a.signum() * (a.abs() - high * w).max(0.)).abs() < 1e-12);
    }
    assert_eq!(theta[3], original[3]);
    assert!((projection::mass(&theta, &weights) - radius).abs() < 1e-12);
    assert!(projection::project(&mut theta, &[1., -1., 1., 1.], radius).is_err());
    Ok(())
}
#[test]
fn ordinary_ce_gradient_and_unequal_original_batches() -> Result<()> {
    let mut model = arbitrary(Bank::Hinges16)?;
    let rows = Encoder::new(model.bank).records(&model.input);
    let all: Vec<_> = (0..rows.len()).collect();
    let (ce, g) = training::gradient(&model, &rows, &all)?;
    let score = fit::score(&model, &rows, &all);
    assert!((ce - score.ce_sum / rows.len() as f64).abs() < 1e-12);
    let mut pooled = vec![0.; g.len()];
    let mut pooled_ce = 0.;
    for batch in all.chunks(7) {
        let (loss, bg) = training::gradient(&model, &rows, batch)?;
        let weight = batch.len() as f64 / rows.len() as f64;
        pooled_ce += loss * weight;
        for (p, b) in pooled.iter_mut().zip(bg) {
            *p += b * weight;
        }
    }
    assert_eq!(all.chunks(7).last().unwrap().len(), 1);
    assert!((pooled_ce - ce).abs() < 1e-11);
    for (a, b) in pooled.iter().zip(&g) {
        assert!((a - b).abs() < 1e-12);
    }
    for (j, &analytic) in g.iter().enumerate() {
        let saved = model.theta[j];
        let h = 1e-4;
        model.theta[j] = saved + h;
        let plus = fit::score(&model, &rows, &all).ce_sum / rows.len() as f64;
        model.theta[j] = saved - h;
        let minus = fit::score(&model, &rows, &all).ce_sum / rows.len() as f64;
        model.theta[j] = saved;
        assert!(
            ((plus - minus) / (2. * h) - analytic).abs() < 2e-8,
            "coordinate{j}"
        );
    }
    Ok(())
}
#[test]
fn exclusion_uses_no_held_labels_and_preserves_categorical_permutations() -> Result<()> {
    let input = fixture();
    let held = 0;
    let train: Vec<_> = (0..input.records.len())
        .filter(|&i| fit::fold(input.records[i].cell) != held)
        .collect();
    let moments = fit::Moments::new(&input, Bank::Hinges16, &train)?;
    let state = moments.solve(RIDGES[1], 1, &moments.factor(RIDGES[1])?)?;
    let mut changed = input.clone();
    for r in &mut changed.records {
        if fit::fold(r.cell) == held {
            r.label = (r.label + 4) % input.vocabulary;
        }
    }
    let moments = fit::Moments::new(&changed, Bank::Hinges16, &train)?;
    assert_eq!(
        state,
        moments.solve(RIDGES[1], 1, &moments.factor(RIDGES[1])?)?
    );
    let permutation = |c: usize| (2 * c + 1) % 9;
    let mut permuted = input.clone();
    for r in &mut permuted.records {
        r.label = permutation(r.label);
    }
    let (a, report_a) = fit::run(&input)?;
    let (b, report_b) = fit::run(&permuted)?;
    assert_eq!(report_a.selected, report_b.selected);
    assert_eq!(a.theta, b.theta);
    for cell in 0..81 {
        let za = a.logits_cell(cell)?;
        let zb = b.logits_cell(cell)?;
        for c in 0..9 {
            assert_eq!(za[c], zb[permutation(c)]);
        }
    }
    for (f, ids) in report_a.fold_record_ids.iter().enumerate() {
        for id in ids {
            assert_eq!(fit::fold(input.records[*id].cell), f);
        }
    }
    Ok(())
}
#[test]
fn shared_moments_predict_new_linear_and_nonlinear_examples() -> Result<()> {
    for nonlinear in [false, true] {
        let n = 25;
        let labels = |cell: usize| {
            let (x, y) = (cell / n, cell % n);
            let b = (x + y) % 3;
            let boundary = if nonlinear {
                12. + 7. * (std::f64::consts::TAU * x as f64 / 24.).sin()
            } else {
                x as f64
            };
            if (y as f64) < boundary + b as f64 - 1. {
                2 * b
            } else {
                2 * b + 1
            }
        };
        let input = Input {
            parent_sha256: "1".repeat(64),
            source_sha256: "2".repeat(64),
            n,
            vocabulary: n,
            blocks: 3,
            route: (0..n * n).map(|cell| (cell / n + cell % n) % 3).collect(),
            records: (0..n * n)
                .filter(|cell| cell % 5 != 0)
                .enumerate()
                .map(|(id, cell)| Record {
                    id,
                    cell,
                    label: labels(cell),
                })
                .collect(),
        };
        let bank = if nonlinear {
            Bank::Hinges32
        } else {
            Bank::Quadratic
        };
        let all: Vec<_> = (0..input.records.len()).collect();
        let moments = fit::Moments::new(&input, bank, &all)?;
        let model = moments
            .solve(1e-6, 1, &moments.factor(1e-6)?)?
            .model(&input, 32.)?;
        let held: Vec<_> = (0..n * n).filter(|cell| cell % 5 == 0).collect();
        let correct = held
            .iter()
            .filter(|&&cell| argmax(&model.logits_cell(cell).unwrap()) == labels(cell))
            .count();
        assert!(
            correct as f64 / held.len() as f64 > 0.85,
            "nonlinear{nonlinear}: {correct}/{}",
            held.len()
        );
    }
    Ok(())
}
#[test]
fn full_mini_forward_tails_and_entire_saved_state_replay() -> Result<()> {
    let input = fixture();
    for mode in [Mode::Full, Mode::Mini, Mode::Forward] {
        let policy = Policy {
            mode,
            iterations: if mode == Mode::Forward { 0 } else { 2 },
            batch: 7,
            learning_rate: 1.,
        };
        let result = training::run(input.clone(), policy)?;
        if mode == Mode::Forward {
            assert_eq!(result.work.gradient_batches, 0);
            assert!(result.steps.is_empty());
        } else {
            assert!(result.work.effective_updates > 0);
        }
        if mode == Mode::Mini {
            assert_eq!(result.steps.last().unwrap().record_ids.len(), 1);
        }
        assert_eq!(result.initialization.ce_gradients, 0);
        assert_eq!(result.initialization.candidates.len(), 432);
        assert_eq!(result.initialization.pooled_scores.len(), 576);
        let saved: training::Outcome = serde_json::from_slice(&serde_json::to_vec(&result)?)?;
        assert_eq!(saved, result);
        training::replay(&input, &saved)?;
        let mut corrupt = saved;
        corrupt.model.theta[0] += 1.;
        assert!(training::replay(&input, &corrupt).is_err());
    }
    Ok(())
}
#[test]
fn resource_forecast_and_unsupported_decoder_refusal() -> Result<()> {
    let input = fixture();
    let a = limits::forecast(&input, &Policy::forward())?;
    assert_eq!(a.candidate_scored_records, 576 * input.records.len());
    assert_eq!(a.candidate_scores, 1728);
    assert_eq!(
        a.implied_complete_coefficients,
        input.blocks * 257 * input.vocabulary
    );
    let mut renamed = input.clone();
    for r in &mut renamed.records {
        r.label = (2 * r.label + 1) % input.vocabulary;
    }
    assert_eq!(a, limits::forecast(&renamed, &Policy::forward())?);
    let mut unsupported = input;
    unsupported.records[0].label = 4;
    let all: Vec<_> = (0..unsupported.records.len()).collect();
    assert!(fit::Moments::new(&unsupported, Bank::Quadratic, &all).is_err());
    Ok(())
}

use super::*;
use crate::features::{Mask, Transport};

fn bank() -> Bank {
    Bank {
        vocab: 3,
        depth: 0,
        history_slots: 8,
        masks: vec![Mask::All],
        transport: Transport::Fourier,
        execution: crate::features::Execution::Composed,
    }
}
fn example(tokens: Vec<i64>, targets: Vec<i64>) -> data::Example {
    data::Example {
        task: "test".into(),
        changes: vec![false; tokens.len()],
        tokens,
        targets,
        prompt: vec![],
        answer: vec![],
        limit: 0,
        field: None,
        row_index: None,
    }
}

#[test]
fn empirical_gradient_matches_finite_differences() -> Result<()> {
    let mut model = Head::zero(bank(), 20.)?;
    for (i, w) in model.coefficients.iter_mut().enumerate() {
        *w = 0.03 * (i as f64 - 10.);
    }
    let cache = Cache::build(
        &model.bank,
        &[
            example(vec![0, 1, 1], vec![0, 2, 1]),
            example(vec![2], vec![0]),
        ],
    )?;
    let order = cache.full_order();
    let mut gradient = vec![0.; model.coefficients.len()];
    objective(&model, &cache, &order, Some(&mut gradient))?;
    for (i, &g) in gradient.iter().enumerate() {
        let old = model.coefficients[i];
        model.coefficients[i] = old + 1e-5;
        let plus = objective(&model, &cache, &order, None)?.0;
        model.coefficients[i] = old - 1e-5;
        let minus = objective(&model, &cache, &order, None)?.0;
        model.coefficients[i] = old;
        assert!((g - (plus - minus) / 2e-5).abs() < 1e-9);
    }
    Ok(())
}

#[test]
fn projection_is_the_l1_euclidean_projection_and_preserves_feasible_heads() -> Result<()> {
    let mut model = Head::zero(bank(), 3.)?;
    model.coefficients[0] = 4.;
    model.coefficients[3] = -2.;
    model.coefficients[6] = 1.;
    model.coefficients[1] = -0.5;
    model.project();
    assert_eq!(model.coefficients[0], 2.5);
    assert_eq!(model.coefficients[3], -0.5);
    assert_eq!(model.coefficients[6], 0.);
    assert_eq!(model.coefficients[1], -0.5);
    let first = model.coefficients.clone();
    model.project();
    assert_eq!(model.coefficients, first);
    assert_eq!(model.validate()?, 3.);
    model.coefficients[0] += 0.01;
    assert!(model.validate().is_err());
    Ok(())
}

#[test]
fn example_minibatches_keep_multiplicity_and_the_short_tail() -> Result<()> {
    let model = Head::zero(bank(), 20.)?;
    let cache = Cache::build(
        &model.bank,
        &[
            example(vec![0, 1, 2], vec![0, 1, 2]),
            example(vec![1, 1], vec![-100, 0]),
            example(vec![2, 2], vec![0, 1]),
        ],
    )?;
    assert_eq!(cache.example_offsets, vec![0, 3, 4, 6]);
    let order = cache.full_order();
    let mut full = vec![0.; model.coefficients.len()];
    let (loss, n) = objective(&model, &cache, &order, Some(&mut full))?;
    let (mut combined_loss, mut combined) = (0., vec![0.; full.len()]);
    let mut part = vec![0.; full.len()];
    for batch in order.chunks(2) {
        let (l, count) = objective(&model, &cache, batch, Some(&mut part))?;
        combined_loss += l * count as f64;
        for (a, b) in combined.iter_mut().zip(&part) {
            *a += b * count as f64;
        }
    }
    assert!((combined_loss / n as f64 - loss).abs() < 1e-14);
    for (a, b) in combined.iter().zip(&full) {
        assert!((a / n as f64 - b).abs() < 1e-14);
    }
    assert!(objective(&model, &cache, &[usize::MAX], None).is_err());
    Ok(())
}

#[test]
fn forward_initialization_fits_empirical_frequencies_not_deduplicated_labels() -> Result<()> {
    let mut model = Head::zero(bank(), 96.)?;
    let rows = [
        example(vec![0], vec![0]),
        example(vec![0], vec![0]),
        example(vec![0], vec![0]),
        example(vec![0], vec![1]),
        example(vec![2], vec![2]),
    ];
    let cache = Cache::build(&model.bank, &rows)?;
    let mut scratch = vec![0.; model.coefficients.len()];
    assert_eq!(forward_initialize(&mut model, &cache, &mut scratch)?, 0);
    let logits = model.logits(&[0]);
    let mut residual = vec![0.; 3];
    ce_gradient(&logits, 2, &mut residual);
    assert!((residual[0] - 0.75).abs() < 1e-10);
    assert!((residual[1] - 0.25).abs() < 1e-10);
    let loss = objective(&model, &cache, &cache.full_order(), None)?.0;
    let expected = -(3. * 0.75f64.ln() + 0.25f64.ln()) / 5.;
    assert!((loss - expected).abs() < 1e-9);
    Ok(())
}

#[test]
fn full_gradient_step_decreases_ce_and_gap_matches_exhaustive_oracle() -> Result<()> {
    let mut model = Head::zero(bank(), 3.)?;
    let cache = Cache::build(&model.bank, &[example(vec![0, 1, 2], vec![1, 2, 0])])?;
    let mut gradient = vec![0.; model.coefficients.len()];
    let order = cache.full_order();
    let initial = objective(&model, &cache, &order, Some(&mut gradient))?.0;
    gradient_step(&mut model, &gradient, 0.5)?;
    assert!(objective(&model, &cache, &order, None)?.0 < initial);
    let gap = linear_gap(&model, &gradient)?;
    let dot: f64 = model
        .coefficients
        .iter()
        .zip(&gradient)
        .map(|(w, g)| w * g)
        .sum();
    let mut minimum = 0.;
    for c in 0..3 {
        let mut best = f64::INFINITY;
        for k in 0..model.bank.dimension() {
            for sign in [-1., 1.] {
                best = best.min(sign * model.radius * gradient[k * 3 + c]);
            }
        }
        minimum += best;
    }
    assert!((gap - dot + minimum).abs() < 1e-14);
    Ok(())
}

#[test]
fn oversized_banks_are_rejected_before_any_coefficients_or_cache_allocate() {
    let large = Bank {
        vocab: 548,
        depth: 4,
        history_slots: 4096,
        masks: vec![
            Mask::All,
            Mask::Pair(1, 2),
            Mask::Pair(1, 3),
            Mask::Pair(1, 4),
            Mask::Pair(2, 3),
            Mask::Pair(2, 4),
            Mask::Pair(3, 4),
        ],
        transport: Transport::Fourier,
        execution: crate::features::Execution::Composed,
    };
    assert!(Head::zero(large.clone(), 100.).is_err());
    assert!(Cache::build(&large, &[example(vec![1], vec![0])]).is_err());
}

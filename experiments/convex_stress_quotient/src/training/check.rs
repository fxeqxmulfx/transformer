use super::*;
fn close(a: f64, b: f64) -> bool {
    a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-9 * (1. + a.abs() + b.abs())
}
/// Independent dense class/latent accumulation; no producer gradient or logits.
pub(crate) fn reference(model: &Model, batch: &[[usize; 2]]) -> (f64, Vec<f64>) {
    let mut gradient = vec![0.; model.coefficients.len()];
    let mut loss = 0.;
    for &[cell, label] in batch {
        let z = (model.rows[cell % model.n] as i128 + model.columns[cell / model.n] as i128)
            .rem_euclid(model.q as i128) as usize;
        let logits = &model.coefficients[z * model.vocabulary..(z + 1) * model.vocabulary];
        let shift = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let masses: Vec<_> = logits.iter().map(|x| (x - shift).exp()).collect();
        let total: f64 = masses.iter().sum();
        loss += total.ln() + shift - logits[label];
        for (c, mass) in masses.iter().enumerate() {
            gradient[z * model.vocabulary + c] += mass / total;
        }
        gradient[z * model.vocabulary + label] -= 1.;
    }
    for g in &mut gradient {
        *g /= batch.len() as f64;
    }
    (loss / batch.len() as f64, gradient)
}
fn reference_score(model: &Model, batch: &[[usize; 2]]) -> f64 {
    // No derivatives in a forward-only check.
    batch
        .iter()
        .map(|&[cell, label]| {
            let z = (model.rows[cell % model.n] as usize + model.columns[cell / model.n] as usize)
                % model.q;
            let w = &model.coefficients[z * model.vocabulary..(z + 1) * model.vocabulary];
            let top = w.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            w.iter().map(|x| (x - top).exp()).sum::<f64>().ln() + top - w[label]
        })
        .sum::<f64>()
        / batch.len() as f64
}
pub fn check(initial: &Model, samples: &[[usize; 2]], out: &Outcome) -> Result<()> {
    initial.observations(samples)?;
    let forecast = out.policy.validate(initial, samples.len())?;
    out.model.observations(samples)?;
    let mut expected = initial.clone();
    let mut work = Work {
        score_examples: samples.len(),
        ..Default::default()
    };
    if out.steps.len() != out.policy.updates
        || out.scalar_work_forecast != forecast
        || !close(out.initial_ce, reference_score(initial, samples))
    {
        return Err("head initial CE, policy length or forecast differs".into());
    }
    for (index, step) in out.steps.iter().enumerate() {
        let count = out.policy.count(samples.len());
        let start = index * count % samples.len();
        let batch: Vec<_> = (0..count)
            .map(|i| samples[(start + i) % samples.len()])
            .collect();
        let (ce, derivative) = reference(&expected, &batch);
        let norm = derivative.iter().map(|x| x * x).sum::<f64>().sqrt();
        for (w, g) in expected.coefficients.iter_mut().zip(derivative) {
            *w -= out.policy.learning_rate * g;
        }
        projection::reference(
            &mut expected.coefficients,
            expected.vocabulary,
            expected.radius,
        );
        expected.validate()?;
        if (step.start, step.count) != (start, count)
            || !close(step.batch_ce, ce)
            || !close(step.gradient_l2, norm)
            || !close(step.post_train_ce, reference_score(&expected, samples))
        {
            return Err(
                "independent CE derivative, batch, projection or step score differs".into(),
            );
        }
        work.gradient_batches += 1;
        work.gradient_examples += count;
        work.score_examples += samples.len();
        work.projection_columns += expected.vocabulary;
        work.coefficient_updates += expected.coefficients.len();
    }
    if expected
        .coefficients
        .iter()
        .zip(&out.model.coefficients)
        .any(|(&a, &b)| !close(a, b))
    {
        return Err("independent projected CE trajectory differs".into());
    }
    expected.coefficients.clone_from(&out.model.coefficients);
    if expected != out.model || work != out.work {
        return Err("fixed feature state or actual work differs".into());
    }
    Ok(())
}

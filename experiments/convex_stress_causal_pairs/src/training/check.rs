//! Independent complex arithmetic and accumulation order, plus bisection L1
//! projection. Exact producer replay separately checks the stored floating bits.
use super::*;
pub(super) fn reference(model: &Model, samples: &[Encoded]) -> (f64, Vec<f64>) {
    let mut residuals = vec![];
    let mut grouped = vec![vec![]; model.blocks.len()];
    let mut loss = 0.;
    for (i, s) in samples.iter().enumerate() {
        grouped[s.block].push(i);
        let b = &model.blocks[s.block];
        let n = b.states();
        let z: Vec<_> = (0..model.vocabulary)
            .map(|c| {
                (0..n)
                    .map(|f| {
                        let row = (b.first_row + 2 * f) * model.vocabulary;
                        (C::new(
                            model.coefficients[row + c],
                            model.coefficients[row + model.vocabulary + c],
                        ) * b.code[s.state * n + f])
                            .re
                    })
                    .sum::<f64>()
            })
            .collect();
        let top = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mass = z.iter().map(|v| (v - top).exp()).sum::<f64>();
        loss += mass.ln() + (top - z[s.label]);
        residuals.push(
            z.iter()
                .enumerate()
                .map(|(c, &v)| (v - top).exp() / mass - f64::from(c == s.label))
                .collect::<Vec<_>>(),
        );
    }
    let mut gradient = vec![0.; model.coefficients.len()];
    for (block, b) in model.blocks.iter().enumerate() {
        for f in 0..b.states() {
            for c in 0..model.vocabulary {
                let mut delta = C::new(0., 0.);
                for &i in &grouped[block] {
                    let s = samples[i];
                    delta += b.code[s.state * b.states() + f].conj() * residuals[i][c];
                }
                let row = (b.first_row + 2 * f) * model.vocabulary;
                gradient[row + c] = delta.re / samples.len() as f64;
                gradient[row + model.vocabulary + c] = delta.im / samples.len() as f64;
            }
        }
    }
    (loss / samples.len() as f64, gradient)
}
fn close(a: f64, b: f64) -> bool {
    a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-9 * (1. + a.abs() + b.abs())
}
pub fn check(initial: &Model, input: &Input, out: &Outcome) -> Result<()> {
    let samples = initial.observations(input)?;
    out.model.validate()?;
    if out.scalar_work_forecast != out.policy.preflight(initial, &samples)?
        || out.steps.len() != out.policy.updates
        || out.model.blocks != initial.blocks
        || out.model.context != initial.context
        || out.model.vocabulary != initial.vocabulary
        || out.model.radius != initial.radius
        || !close(out.initial_ce, score(initial, &samples))
    {
        return Err("complex training policy, fixed features or initial CE changed".into());
    }
    let mut model = initial.clone();
    let mut work = Work {
        score_examples: samples.len(),
        ..Work::default()
    };
    for (step, saved) in out.steps.iter().enumerate() {
        let count = out.policy.count(samples.len());
        let start = step * count % samples.len();
        let batch: Vec<_> = (0..count)
            .map(|i| samples[(start + i) % samples.len()])
            .collect();
        let (loss, g) = reference(&model, &batch);
        let norm = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        if saved.start != start
            || saved.count != count
            || !close(saved.batch_ce, loss)
            || !close(saved.gradient_l2, norm)
        {
            return Err("independent ordinary complex CE derivative/trajectory differs".into());
        }
        for (x, d) in model.coefficients.iter_mut().zip(g) {
            *x -= out.policy.learning_rate * d;
        }
        projection::reference(&mut model.coefficients, model.vocabulary, model.radius);
        if !close(saved.post_batch_ce, score(&model, &batch)) {
            return Err("independent post-batch CE differs".into());
        }
        work.gradient_batches += 1;
        work.gradient_examples += count;
        work.score_examples += count;
        work.scalar_derivative_terms += batch
            .iter()
            .map(|s| 2 * model.blocks[s.block].states() * model.vocabulary)
            .sum::<usize>();
        work.projection_columns += model.vocabulary;
        work.coefficient_updates += model.coefficients.len();
    }
    work.score_examples += samples.len();
    if work != out.work
        || !close(out.final_ce, score(&model, &samples))
        || model
            .coefficients
            .iter()
            .zip(&out.model.coefficients)
            .any(|(&a, &b)| !close(a, b))
    {
        return Err("independent final complex head, raw CE or work differs".into());
    }
    Ok(())
}

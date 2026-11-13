//! Separate complex-arithmetic derivatives and projected state reconstruction.
//! Every independently recomputed iterate is compared with the native state
//! whose table/coefficient hashes are saved in the trajectory.
use super::*;
fn near(a: f64, b: f64) -> bool {
    audit::close(a, b, 1e-8)
}
fn reference_score(model: &Model, batch: &[Encoded]) -> f64 {
    batch
        .iter()
        .map(|s| {
            let z = audit::complex_logits(model, (s.block, s.state));
            let shift = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            z.iter().map(|x| (x - shift).exp()).sum::<f64>().ln() + shift - z[s.label]
        })
        .sum::<f64>()
        / batch.len() as f64
}
pub(crate) fn reference(model: &Model, batch: &[Encoded]) -> Gradient {
    let v = model.vocabulary;
    let mut complex = vec![C::new(0., 0.); model.table.len()];
    for s in batch {
        let b = &model.blocks[s.block];
        let logits = audit::complex_logits(model, (s.block, s.state));
        let shift = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let p: Vec<_> = logits.iter().map(|x| (x - shift).exp()).collect();
        let total: f64 = p.iter().sum();
        for f in 0..b.states() {
            let conjugate = b.code[s.state * b.states() + f].conj();
            for (c, &mass) in p.iter().enumerate() {
                complex[(b.offset + f) * v + c] +=
                    conjugate * (mass / total - f64::from(c == s.label));
            }
        }
    }
    for g in &mut complex {
        *g /= batch.len() as f64;
    }
    let mut head = vec![0.; model.coefficients.len()];
    let mut table = vec![0.; model.table.len()];
    for b in &model.blocks {
        for s in 0..b.states() {
            for c in 0..v {
                let g = complex[(b.offset + s) * v + c];
                head[2 * (b.offset + s) * v + c] = g.re;
                head[(2 * (b.offset + s) + 1) * v + c] = g.im;
                table[(b.offset + s) * v + c] = (0..b.states())
                    .map(|f| b.code[s * b.states() + f] * complex[(b.offset + f) * v + c])
                    .sum::<C>()
                    .re;
            }
        }
    }
    Gradient {
        loss: reference_score(model, batch),
        head,
        table,
    }
}
fn reference_step(model: &mut Model, g: &Gradient, rate: f64) {
    for b in &model.blocks {
        for s in 0..b.states() {
            for c in 0..model.vocabulary {
                let k = (b.offset + s) * model.vocabulary + c;
                model.table[k] = if s + 1 == b.states() {
                    f64::from(c == b.default_label) * 24.
                } else {
                    (-32_f64).max(32_f64.min(model.table[k] - rate * g.table[k]))
                };
            }
        }
    }
    model.coefficients = audit::transformed(model, &model.table);
}
pub fn check(initial: &Model, samples: &[[usize; 2]], out: &Outcome) -> Result<()> {
    initial.validate()?;
    out.model.validate()?;
    let encoded = initial.encode(samples)?;
    let forecast = out.policy.preflight(initial, &encoded)?;
    if forecast != out.scalar_work_forecast
        || out.steps.len() != out.policy.updates
        || !near(out.initial_ce, reference_score(initial, &encoded))
    {
        return Err("refined initial loss, trajectory length or forecast differs".into());
    }
    let mut independent = initial.clone();
    let mut native = initial.clone();
    let mut work = Work {
        score_examples: samples.len(),
        ..Default::default()
    };
    for (index, saved) in out.steps.iter().enumerate() {
        let (start, count) = out.policy.span(samples.len(), index);
        let batch = &encoded[start..start + count];
        let g = reference(&independent, batch);
        let actual = gradient(&native, batch)?;
        if !near(g.loss, actual.loss)
            || g.head.iter().zip(&actual.head).any(|(&a, &b)| !near(a, b))
            || g.table
                .iter()
                .zip(&actual.table)
                .any(|(&a, &b)| !near(a, b))
        {
            return Err("actual real/imaginary derivative or table pullback differs".into());
        }
        reference_step(&mut independent, &g, out.policy.learning_rate);
        independent.validate()?;
        let step = advance(&mut native, batch, start, out.policy.learning_rate)?;
        if step != *saved
            || !near(saved.batch_ce, g.loss)
            || !near(saved.head_gradient_l2, norm(&g.head))
            || !near(saved.table_gradient_l2, norm(&g.table))
            || !near(saved.post_batch_ce, reference_score(&independent, batch))
            || native
                .table
                .iter()
                .zip(&independent.table)
                .any(|(&a, &b)| !near(a, b))
            || native
                .coefficients
                .iter()
                .zip(&independent.coefficients)
                .any(|(&a, &b)| !near(a, b))
        {
            return Err("saved hashes or independent projected complex iterate differs".into());
        }
        charge(&native, batch, &mut work);
    }
    work.score_examples += samples.len();
    if out.work != work
        || out.model != native
        || !near(out.final_ce, reference_score(&independent, &encoded))
    {
        return Err("refined endpoint, fixed extractor, work or final CE differs".into());
    }
    Ok(())
}

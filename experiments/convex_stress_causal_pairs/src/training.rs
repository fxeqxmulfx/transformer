//! CD8 projected ordinary CE on the actual real/imaginary Fourier head.
//! Batch losses are retained per update; full-training CE is measured at both
//! endpoints. The mini path does not rescore all training prefixes every update.
use crate::{
    model::{Encoded, Model},
    *,
};
mod check;
mod projection;
#[cfg(test)]
mod tests;
pub use check::check;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Mode {
    Full,
    Mini,
    Forward,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Policy {
    pub mode: Mode,
    pub updates: usize,
    pub batch: usize,
    pub learning_rate: f64,
}
pub fn policies() -> [Policy; 3] {
    [
        Policy {
            mode: Mode::Full,
            updates: 16,
            batch: 64,
            learning_rate: 1.,
        },
        Policy {
            mode: Mode::Mini,
            updates: 256,
            batch: 64,
            learning_rate: 0.1,
        },
        Policy {
            mode: Mode::Forward,
            updates: 0,
            batch: 64,
            learning_rate: 1.,
        },
    ]
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Work {
    pub gradient_batches: usize,
    pub gradient_examples: usize,
    pub score_examples: usize,
    pub scalar_derivative_terms: usize,
    pub projection_columns: usize,
    pub coefficient_updates: usize,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Step {
    pub start: usize,
    pub count: usize,
    pub batch_ce: f64,
    pub gradient_l2: f64,
    pub post_batch_ce: f64,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Outcome {
    pub policy: Policy,
    pub initial_ce: f64,
    pub final_ce: f64,
    pub model: Model,
    pub steps: Vec<Step>,
    pub work: Work,
    pub scalar_work_forecast: u64,
}
impl Policy {
    fn count(&self, n: usize) -> usize {
        if self.mode == Mode::Full {
            n
        } else {
            self.batch.min(n)
        }
    }
    fn preflight(&self, model: &Model, samples: &[Encoded]) -> Result<u64> {
        if self.updates > 4096
            || !(1..=65536).contains(&self.batch)
            || !self.learning_rate.is_finite()
            || self.learning_rate <= 0.
            || self.learning_rate > 1.
            || (self.mode == Mode::Forward) != (self.updates == 0)
        {
            return Err("invalid bounded complex CE policy".into());
        }
        let terms = |s: &Encoded| 2 * model.blocks[s.block].states() * model.vocabulary;
        let all = samples.iter().map(terms).sum::<usize>();
        let batch = if self.mode == Mode::Full {
            all
        } else {
            self.count(samples.len()) * samples.iter().map(terms).max().unwrap_or(0)
        };
        let work = 4 * all as u64
            + self.updates as u64 * (8 * batch as u64 + 64 * model.coefficients.len() as u64);
        if work > 2_000_000_000 {
            return Err("complex CE training exceeds2B scalar-work forecast".into());
        }
        Ok(work)
    }
}
pub fn ce(z: &[f64], label: usize) -> f64 {
    let top = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    z.iter().map(|x| (x - top).exp()).sum::<f64>().ln() + (top - z[label])
}
pub fn score(model: &Model, samples: &[Encoded]) -> f64 {
    samples
        .iter()
        .map(|s| ce(&model.logits_state(s.block, s.state), s.label))
        .sum::<f64>()
        / samples.len() as f64
}
pub fn gradient(model: &Model, samples: &[Encoded]) -> Result<(f64, Vec<f64>)> {
    if samples.is_empty()
        || samples.iter().any(|s| {
            s.block >= model.blocks.len()
                || s.state >= model.blocks[s.block].states()
                || s.label >= model.vocabulary
        })
    {
        return Err("invalid actual complex gradient sample".into());
    }
    let mut out = vec![0.; model.coefficients.len()];
    let mut loss = 0.;
    for s in samples {
        let b = &model.blocks[s.block];
        let n = b.states();
        let z = model.logits_state(s.block, s.state);
        loss += ce(&z, s.label);
        let top = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mass = z.iter().map(|x| (x - top).exp()).sum::<f64>();
        for (c, &logit) in z.iter().enumerate() {
            let delta =
                ((logit - top).exp() / mass - f64::from(c == s.label)) / samples.len() as f64;
            for f in 0..n {
                let phi = b.code[s.state * n + f];
                let row = (b.first_row + 2 * f) * model.vocabulary;
                out[row + c] += delta * phi.re;
                out[row + model.vocabulary + c] -= delta * phi.im;
            }
        }
    }
    Ok((loss / samples.len() as f64, out))
}
pub fn run(initial: &Model, input: &Input, policy: Policy) -> Result<Outcome> {
    let samples = initial.observations(input)?;
    let forecast = policy.preflight(initial, &samples)?;
    let mut model = initial.clone();
    let initial_ce = score(&model, &samples);
    let mut work = Work {
        score_examples: samples.len(),
        ..Work::default()
    };
    let mut steps = vec![];
    for step in 0..policy.updates {
        let count = policy.count(samples.len());
        let start = step * count % samples.len();
        let batch: Vec<_> = (0..count)
            .map(|i| samples[(start + i) % samples.len()])
            .collect();
        let (batch_ce, g) = gradient(&model, &batch)?;
        let gradient_l2 = g.iter().map(|x| x * x).sum::<f64>().sqrt();
        for (w, d) in model.coefficients.iter_mut().zip(g) {
            *w -= policy.learning_rate * d;
        }
        projection::project(&mut model.coefficients, model.vocabulary, model.radius);
        model.check_coefficients()?;
        let post_batch_ce = score(&model, &batch);
        steps.push(Step {
            start,
            count,
            batch_ce,
            gradient_l2,
            post_batch_ce,
        });
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
    let final_ce = score(&model, &samples);
    work.score_examples += samples.len();
    Ok(Outcome {
        policy,
        initial_ce,
        final_ce,
        model,
        steps,
        work,
        scalar_work_forecast: forecast,
    })
}
pub fn replay(initial: &Model, input: &Input, out: &Outcome) -> Result<()> {
    check(initial, input, out)?;
    if run(initial, input, out.policy.clone())? != *out {
        return Err("saved complex CE states, batches or work fail exact replay".into());
    }
    Ok(())
}

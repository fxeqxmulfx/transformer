//! CD8: projected ordinary cross-entropy on one fixed latent feature table.
use crate::{Result, model::Model};
use serde::{Deserialize, Serialize};
mod check;
mod projection;
pub use check::check;
use projection::project;
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
    pub projection_columns: usize,
    pub coefficient_updates: usize,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Step {
    pub start: usize,
    pub count: usize,
    pub batch_ce: f64,
    pub gradient_l2: f64,
    pub post_train_ce: f64,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Outcome {
    pub policy: Policy,
    pub initial_ce: f64,
    pub model: Model,
    pub steps: Vec<Step>,
    pub work: Work,
    pub scalar_work_forecast: u64,
}
impl Policy {
    fn validate(&self, model: &Model, count: usize) -> Result<u64> {
        self.preflight(model.n, model.q, model.vocabulary, count)
    }
    pub(crate) fn preflight(
        &self,
        n: usize,
        q: usize,
        vocabulary: usize,
        count: usize,
    ) -> Result<u64> {
        if self.updates > 4096
            || !(2..=257).contains(&n)
            || !(2..=257).contains(&q)
            || !(2..=1024).contains(&vocabulary)
            || !(1..=257 * 257).contains(&count)
            || !(1..=65536).contains(&self.batch)
            || !self.learning_rate.is_finite()
            || self.learning_rate <= 0.
            || self.learning_rate > 1.
            || (self.mode == Mode::Forward) != (self.updates == 0)
        {
            return Err("invalid bounded head training policy".into());
        }
        let batch = self.count(count);
        let forecast = (count as u64 * vocabulary as u64)
            + 2 * (n * n) as u64
            + self.updates as u64
                * (4 * (count + batch) as u64 * vocabulary as u64
                    + 64 * (q * vocabulary) as u64
                    + 2 * (n * n) as u64);
        if forecast > 2_000_000_000 {
            return Err("head training scalar-work forecast exceeds2B".into());
        }
        Ok(forecast)
    }
    fn count(&self, count: usize) -> usize {
        if self.mode == Mode::Full {
            count
        } else {
            self.batch.min(count)
        }
    }
}
pub fn ce(logits: &[f64], label: usize) -> f64 {
    let top = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    logits.iter().map(|x| (x - top).exp()).sum::<f64>().ln() + (top - logits[label])
}
pub fn score(model: &Model, samples: &[[usize; 2]]) -> f64 {
    samples
        .iter()
        .map(|&[cell, label]| ce(model.logits(cell), label))
        .sum::<f64>()
        / samples.len() as f64
}
/// Mean CE derivative with respect to the actual combined W[z,c] tensor.
pub fn gradient(model: &Model, samples: &[[usize; 2]]) -> Result<(f64, Vec<f64>)> {
    if samples.is_empty()
        || samples
            .iter()
            .any(|&[cell, label]| cell >= model.latent.len() || label >= model.vocabulary)
    {
        return Err("invalid CE derivative samples".into());
    }
    let mut derivative = vec![0.; model.coefficients.len()];
    let mut loss = 0.;
    for &[cell, label] in samples {
        let logits = model.logits(cell);
        let top = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mass: f64 = logits.iter().map(|x| (x - top).exp()).sum();
        loss += mass.ln() + (top - logits[label]);
        let offset = model.latent[cell] as usize * model.vocabulary;
        for (c, &x) in logits.iter().enumerate() {
            derivative[offset + c] +=
                ((x - top).exp() / mass - f64::from(c == label)) / samples.len() as f64;
        }
    }
    Ok((loss / samples.len() as f64, derivative))
}
pub fn run(initial: &Model, samples: &[[usize; 2]], policy: Policy) -> Result<Outcome> {
    initial.observations(samples)?;
    let forecast = policy.validate(initial, samples.len())?;
    let mut model = initial.clone();
    let initial_ce = score(&model, samples);
    let mut work = Work {
        score_examples: samples.len(),
        ..Default::default()
    };
    let mut steps = vec![];
    for step in 0..policy.updates {
        let count = policy.count(samples.len());
        let start = step * count % samples.len();
        let batch: Vec<_> = (0..count)
            .map(|i| samples[(start + i) % samples.len()])
            .collect();
        let (batch_ce, derivative) = gradient(&model, &batch)?;
        let gradient_l2 = derivative.iter().map(|x| x * x).sum::<f64>().sqrt();
        for (w, g) in model.coefficients.iter_mut().zip(&derivative) {
            *w -= policy.learning_rate * g;
        }
        project(&mut model.coefficients, model.vocabulary, model.radius);
        model.validate()?;
        let post_train_ce = score(&model, samples);
        steps.push(Step {
            start,
            count,
            batch_ce,
            gradient_l2,
            post_train_ce,
        });
        work.gradient_batches += 1;
        work.gradient_examples += count;
        work.score_examples += samples.len();
        work.projection_columns += model.vocabulary;
        work.coefficient_updates += model.coefficients.len();
    }
    Ok(Outcome {
        policy,
        initial_ce,
        model,
        steps,
        work,
        scalar_work_forecast: forecast,
    })
}
pub fn replay(initial: &Model, samples: &[[usize; 2]], out: &Outcome) -> Result<()> {
    if run(initial, samples, out.policy.clone())? != *out {
        return Err("full saved head states, ordinary CE or work fail deterministic replay".into());
    }
    Ok(())
}

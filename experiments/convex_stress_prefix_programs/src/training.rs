//! Ordinary CE on the actual CM53 affine joint-head slice. Mini iterations are epochs with
//! distinct original record IDs and a real short tail, never cyclic wrapping.
use crate::{
    model::{Encoded, Model},
    *,
};
#[path = "../../convex_stress_latent_programs/src/training/check.rs"]
mod check;
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
    pub iterations: usize,
    pub batch: usize,
    pub learning_rate: f64,
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Work {
    pub gradient_batches: usize,
    pub gradient_examples: usize,
    pub derivative_terms: u64,
    pub score_examples: usize,
    pub complete_head_products: u64,
    pub cache_products: u64,
    pub projected_steps: usize,
    pub effective_updates: usize,
    pub encoded_programs: usize,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Step {
    pub epoch: usize,
    pub start: usize,
    pub count: usize,
    pub batch_ce: f64,
    pub post_batch_ce: f64,
    pub gradient_l2: f64,
    pub weights: Vec<f64>,
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

pub fn policies() -> Result<[Policy; 3]> {
    Ok(serde_json::from_value(
        serde_json::from_str::<serde_json::Value>(PROTOCOL)?["training"].clone(),
    )?)
}
impl Policy {
    fn schedule(&self, n: usize) -> Result<Vec<(usize, usize, usize)>> {
        if n == 0
            || self.iterations > 64
            || self.batch == 0
            || self.batch > 65536
            || !self.learning_rate.is_finite()
            || self.learning_rate <= 0.
            || self.learning_rate > 1.
            || (self.mode == Mode::Forward) != (self.iterations == 0)
        {
            return Err("invalid bounded ordinary CE policy".into());
        }
        Ok((0..self.iterations)
            .flat_map(|epoch| match self.mode {
                Mode::Full => vec![(epoch, 0, n)],
                Mode::Mini => (0..n)
                    .step_by(self.batch)
                    .map(|start| (epoch, start, self.batch.min(n - start)))
                    .collect(),
                Mode::Forward => vec![],
            })
            .collect())
    }
    /// Dimensions only: admit all complete products before model fitting.
    pub fn forecast(&self, programs: usize, vocabulary: usize, records: usize) -> Result<u64> {
        if !(1..=128).contains(&programs) || !(2..=240).contains(&vocabulary) || records > 200_000 {
            return Err("affine CE dimensions refused".into());
        }
        let schedule = self.schedule(records)?;
        let n = vocabulary as u64;
        let s = STATES as u64;
        let k = programs as u64;
        let gradient = schedule.iter().map(|s| s.2 as u64).sum::<u64>();
        let calls = 2 * records as u64 + 2 * gradient;
        let cache = if gradient > 0 { 16 * k * s * s * n } else { 0 };
        let work = cache
            + calls * (10 * k * s * n + 32 * n)
            + gradient * 4 * k * n
            + schedule.len() as u64 * (32 * k * k + 8 * k * s * n);
        if work > 96_000_000_000 {
            return Err("latent affine CE exceeds96B scalar-work forecast".into());
        }
        Ok(work)
    }
}

pub fn score(model: &Model, samples: &[Encoded]) -> f64 {
    samples
        .iter()
        .map(|s| ce(&model.logits_states(&s.states), s.label))
        .sum::<f64>()
        / samples.len() as f64
}
/// The cache contains actual full products for every latent state and block. The
/// derivative still uses the full mixed complex logits and every output class.
pub fn gradient(
    model: &Model,
    samples: &[Encoded],
    table: &[Vec<Vec<f64>>],
) -> Result<(f64, Vec<f64>)> {
    let n = model.head.vocabulary;
    if samples.is_empty()
        || table.len() != model.programs.len()
        || table
            .iter()
            .any(|b| b.len() != STATES || b.iter().any(|r| r.len() != n))
        || samples.iter().any(|s| {
            s.label >= n
                || s.states.len() != model.programs.len()
                || s.states.iter().any(|&v| v >= STATES)
        })
    {
        return Err("invalid complete mixture gradient batch".into());
    }
    let mut loss = 0.;
    let mut gradient = vec![0.; model.programs.len()];
    for sample in samples {
        let z = model.logits_states(&sample.states);
        loss += ce(&z, sample.label);
        let top = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mass = z.iter().map(|v| (v - top).exp()).sum::<f64>();
        for (c, &value) in z.iter().enumerate() {
            let residual =
                ((value - top).exp() / mass - f64::from(c == sample.label)) / samples.len() as f64;
            for (j, &state) in sample.states.iter().enumerate() {
                gradient[j] += residual * table[j][state][c];
            }
        }
    }
    Ok((loss / samples.len() as f64, gradient))
}

pub fn project(values: &mut [f64]) -> Result<()> {
    if values.is_empty() || values.iter().any(|x| !x.is_finite()) {
        return Err("nonfinite simplex proposal".into());
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let mut total = 0.;
    let mut theta = 0.;
    for (i, &v) in sorted.iter().enumerate() {
        total += v;
        let candidate = (total - 1.) / (i + 1) as f64;
        if v > candidate {
            theta = candidate;
        }
    }
    for v in values.iter_mut() {
        *v = (*v - theta).max(0.);
    }
    let correction = 1. - values.iter().sum::<f64>();
    let best = argmax(values);
    values[best] += correction;
    if values.iter().any(|v| !v.is_finite() || *v < 0.) {
        return Err("simplex projection failed".into());
    }
    Ok(())
}

pub fn run(initial: &Model, input: &Input, policy: Policy) -> Result<Outcome> {
    let samples = initial.encode(input)?;
    let schedule = policy.schedule(samples.len())?;
    let forecast = policy.forecast(
        initial.programs.len(),
        initial.head.vocabulary,
        samples.len(),
    )?;
    let n = initial.head.vocabulary;
    let table: Vec<_> = if schedule.is_empty() {
        vec![]
    } else {
        (0..initial.programs.len())
            .map(|j| {
                (0..STATES)
                    .map(|s| initial.head.block_logits(j, s))
                    .collect()
            })
            .collect()
    };
    let mut model = initial.clone();
    let initial_ce = score(&model, &samples);
    let mut work = Work {
        score_examples: samples.len(),
        complete_head_products: (samples.len() * model.programs.len() * STATES * n) as u64,
        cache_products: if schedule.is_empty() {
            0
        } else {
            (model.programs.len() * STATES * STATES * n) as u64
        },
        encoded_programs: samples.len() * model.programs.len(),
        ..Work::default()
    };
    let mut steps = vec![];
    for (epoch, start, count) in schedule {
        let batch = &samples[start..start + count];
        let (batch_ce, g) = gradient(&model, batch, &table)?;
        let gradient_l2 = g.iter().map(|x| x * x).sum::<f64>().sqrt();
        let previous = model.weights.clone();
        for (w, d) in model.weights.iter_mut().zip(g) {
            *w -= policy.learning_rate * d;
        }
        project(&mut model.weights)?;
        model.check_weights()?;
        let post_batch_ce = score(&model, batch);
        steps.push(Step {
            epoch,
            start,
            count,
            batch_ce,
            post_batch_ce,
            gradient_l2,
            weights: model.weights.clone(),
        });
        work.gradient_batches += 1;
        work.gradient_examples += count;
        work.derivative_terms += (count * model.programs.len() * n) as u64;
        work.score_examples += count;
        work.complete_head_products += (2 * count * model.programs.len() * STATES * n) as u64;
        work.projected_steps += 1;
        work.effective_updates += usize::from(previous != model.weights);
    }
    let final_ce = score(&model, &samples);
    work.score_examples += samples.len();
    work.complete_head_products += (samples.len() * model.programs.len() * STATES * n) as u64;
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

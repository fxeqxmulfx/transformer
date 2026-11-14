//! Ported kernel/training.rs at ec5f839d for CM40's fixed complex head.
//! Actual record IDs replace distinct-cell encoding; repeated input records
//! retain their denominator and weight. The real/imaginary derivative and
//! projected update are unchanged. Defaults are one full update, one original
//! minibatch epoch including its tail, or zero CE derivatives in forward mode.
use crate::*;
use kernel::model::Encoded;
use records::Record;
use sha2::{Digest, Sha256};
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
    pub updates: usize,
    pub batch: usize,
    pub learning_rate: f64,
}
pub fn policies(samples: usize, batch: usize) -> Result<[Policy; 3]> {
    if samples == 0 || samples > 257 * 257 || !(1..=65536).contains(&batch) {
        return Err("invalid original minibatch/observation count".into());
    }
    Ok([
        Policy {
            mode: Mode::Full,
            updates: 1,
            batch,
            learning_rate: 1.,
        },
        Policy {
            mode: Mode::Mini,
            updates: samples.div_ceil(batch),
            batch,
            learning_rate: 1.,
        },
        Policy {
            mode: Mode::Forward,
            updates: 0,
            batch,
            learning_rate: 1.,
        },
    ])
}
impl Policy {
    fn span(&self, count: usize, step: usize) -> (usize, usize) {
        if self.mode == Mode::Full {
            return (0, count);
        }
        let start = (step % count.div_ceil(self.batch)) * self.batch;
        (start, self.batch.min(count - start))
    }
    pub(crate) fn preflight(&self, model: &Model, samples: &[Encoded]) -> Result<u64> {
        if self.updates > 4096
            || !(1..=65536).contains(&self.batch)
            || !self.learning_rate.is_finite()
            || self.learning_rate <= 0.
            || self.learning_rate > 1.
            || (self.mode == Mode::Forward) != (self.updates == 0)
        {
            return Err("invalid refined CE policy".into());
        }
        let b = model.bounds()?;
        let terms = |s: &Encoded| (model.blocks[s.block].states() * model.vocabulary) as u64;
        let all: u64 = samples.iter().map(terms).sum();
        let encoding_work = (model.n * model.n) as u64
            * (128 * (model.encoding.code.dimension + 1)
                + 8 * (model.encoding.code.names.rows.len()
                    + model.encoding.code.names.columns.len())) as u64;
        let mut work = 8 * all + encoding_work;
        for step in 0..self.updates {
            let (start, count) = self.span(samples.len(), step);
            let batch: u64 = samples[start..start + count].iter().map(terms).sum();
            work += 16 * batch
                + 16 * (b.phase_entries * model.vocabulary) as u64
                + 32 * b.real_coefficients as u64
                + encoding_work;
        }
        if work > limits::WORK_CAP {
            return Err("refined CE training exceeds32B scalar-work forecast".into());
        }
        Ok(work)
    }
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Work {
    pub gradient_batches: usize,
    pub gradient_examples: usize,
    pub score_examples: usize,
    pub head_derivative_terms: usize,
    pub table_pullback_terms: usize,
    pub transform_terms: usize,
    pub table_projection_entries: usize,
    pub coefficients_written: usize,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Step {
    pub start: usize,
    pub count: usize,
    pub batch_ce: f64,
    pub head_gradient_l2: f64,
    pub table_gradient_l2: f64,
    pub post_batch_ce: f64,
    pub table_sha256: String,
    pub coefficients_sha256: String,
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
pub struct Gradient {
    pub loss: f64,
    pub head: Vec<f64>,
    pub table: Vec<f64>,
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
/// Real/imaginary head derivatives, pulled back through H(T)=U* T. The
/// numerical Gram is not silently replaced by an exact identity matrix.
pub fn gradient(model: &Model, samples: &[Encoded]) -> Result<Gradient> {
    if samples.is_empty()
        || samples.len() > records::MAX_RECORDS
        || samples.iter().any(|s| {
            s.block >= model.blocks.len()
                || model.blocks[s.block].observed_labels.get(s.state) != Some(&s.label)
                || s.label >= model.vocabulary
        })
    {
        return Err("invalid refined derivative batch".into());
    }
    let mut head = vec![0.; model.coefficients.len()];
    let mut loss = 0.;
    for s in samples {
        let b = &model.blocks[s.block];
        let z = model.logits_state(s.block, s.state);
        loss += ce(&z, s.label);
        let top = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let total: f64 = z.iter().map(|x| (x - top).exp()).sum();
        for (c, &logit) in z.iter().enumerate() {
            let delta =
                ((logit - top).exp() / total - f64::from(c == s.label)) / samples.len() as f64;
            for f in 0..b.states() {
                let p = b.code[s.state * b.states() + f];
                let h = 2 * (b.offset + f) * model.vocabulary;
                head[h + c] += p.re * delta;
                head[h + model.vocabulary + c] -= p.im * delta;
            }
        }
    }
    let mut table = vec![0.; model.table.len()];
    for b in &model.blocks {
        for s in 0..b.states() {
            for f in 0..b.states() {
                let p = b.code[s * b.states() + f];
                let h = 2 * (b.offset + f) * model.vocabulary;
                let t = (b.offset + s) * model.vocabulary;
                for c in 0..model.vocabulary {
                    table[t + c] += p.re * head[h + c] - p.im * head[h + model.vocabulary + c];
                }
            }
        }
    }
    Ok(Gradient {
        loss: loss / samples.len() as f64,
        head,
        table,
    })
}
fn digest(values: &[f64]) -> String {
    let mut h = Sha256::new();
    h.update((values.len() as u64).to_le_bytes());
    for value in values {
        h.update(value.to_bits().to_le_bytes());
    }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}
fn norm(values: &[f64]) -> f64 {
    values.iter().map(|x| x * x).sum::<f64>().sqrt()
}
fn advance(model: &mut Model, batch: &[Encoded], start: usize, rate: f64) -> Result<Step> {
    let g = gradient(model, batch)?;
    for (t, d) in model.table.iter_mut().zip(&g.table) {
        *t -= rate * d;
    }
    model.project();
    model.validate()?;
    Ok(Step {
        start,
        count: batch.len(),
        batch_ce: g.loss,
        head_gradient_l2: norm(&g.head),
        table_gradient_l2: norm(&g.table),
        post_batch_ce: score(model, batch),
        table_sha256: digest(&model.table),
        coefficients_sha256: digest(&model.coefficients),
    })
}
fn charge(model: &Model, batch: &[Encoded], work: &mut Work) {
    let phases: usize = model.blocks.iter().map(|b| b.code.len()).sum();
    work.gradient_batches += 1;
    work.gradient_examples += batch.len();
    work.score_examples += batch.len();
    work.head_derivative_terms += batch
        .iter()
        .map(|s| 2 * model.blocks[s.block].states() * model.vocabulary)
        .sum::<usize>();
    work.table_pullback_terms += 2 * phases * model.vocabulary;
    work.transform_terms += 2 * phases * model.vocabulary;
    work.table_projection_entries += model.table.len();
    work.coefficients_written += model.coefficients.len();
}
pub fn run(initial: &Model, samples: &[Record], policy: Policy) -> Result<Outcome> {
    initial.validate()?;
    let encoded = records::encode(initial, samples)?;
    let scalar_work_forecast = policy.preflight(initial, &encoded)?;
    let mut model = initial.clone();
    let initial_ce = score(&model, &encoded);
    let mut work = Work {
        score_examples: samples.len(),
        ..Default::default()
    };
    let mut steps = vec![];
    for step in 0..policy.updates {
        let (start, count) = policy.span(samples.len(), step);
        let batch = &encoded[start..start + count];
        steps.push(advance(&mut model, batch, start, policy.learning_rate)?);
        charge(&model, batch, &mut work);
    }
    let final_ce = score(&model, &encoded);
    work.score_examples += samples.len();
    Ok(Outcome {
        policy,
        initial_ce,
        final_ce,
        model,
        steps,
        work,
        scalar_work_forecast,
    })
}
pub fn replay(initial: &Model, samples: &[Record], out: &Outcome) -> Result<()> {
    check(initial, samples, out)?;
    if run(initial, samples, out.policy.clone())? != *out {
        return Err("complete actual CE trajectory, states or work differs".into());
    }
    Ok(())
}

//! Exact port of kernel/training.rs Policy::preflight at166bebc3 (private in
//! the frozen dependency). A refusal precedes updates; successful forecasts
//! must equal the unchanged native trainer's independently recorded forecast.
use crate::*;
use kernel::model::Encoded;
pub fn work(policy: &training::Policy, model: &Model, samples: &[Encoded]) -> Result<u64> {
    if policy.updates > 4096
        || !(1..=65536).contains(&policy.batch)
        || !policy.learning_rate.is_finite()
        || policy.learning_rate <= 0.
        || policy.learning_rate > 1.
        || (policy.mode == training::Mode::Forward) != (policy.updates == 0)
    {
        return Err("invalid refined CE policy".into());
    }
    let b = model.bounds()?;
    let terms = |s: &Encoded| (model.blocks[s.block].states() * model.vocabulary) as u64;
    let all: u64 = samples.iter().map(terms).sum();
    let encoding_work = (model.n * model.n) as u64
        * (128 * (model.encoding.code.dimension + 1)
            + 8 * (model.encoding.code.names.rows.len() + model.encoding.code.names.columns.len()))
            as u64;
    let mut work = 8 * all + encoding_work;
    for step in 0..policy.updates {
        let (start, count) = if policy.mode == training::Mode::Full {
            (0, samples.len())
        } else {
            let start = (step % samples.len().div_ceil(policy.batch)) * policy.batch;
            (start, policy.batch.min(samples.len() - start))
        };
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

//! Forecast constructor, initializer, full products, gradients and L1
//! projections before allocation. Caller must add source construction/scoring.
use crate::*;
pub const WORK_LIMIT: u128 = 1_000_000_000_000;
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Bounds {
    pub coefficients: usize,
    pub gradient_batches: usize,
    pub initialization_work: u64,
    pub training_work: u64,
    pub total_kernel_work: u64,
    pub buffer_bytes: u64,
    pub prediction_work: u64,
}
pub fn forecast(input: &Input, policy: &training::Policy) -> Result<Bounds> {
    input.validate()?;
    policy.validate()?;
    let n = input.records.len() as u128;
    let b = input.blocks as u128;
    let v = input.vocabulary as u128;
    let d = b * WIDTH as u128 * v;
    let batches = match policy.mode {
        training::Mode::Full => policy.iterations,
        training::Mode::Mini => policy.iterations * input.records.len().div_ceil(policy.batch),
        training::Mode::Forward => 0,
    };
    if batches > 4096 {
        return Err("shared-head gradient batch allowance exceeded".into());
    }
    let init = 4096 * n + 256 * b * v + 512 * d;
    // Every epoch includes a fresh full-training CE; projection charges all
    // columns and a conservative full sort even when already inside the ball.
    let epochs = policy.iterations as u128;
    let train = 32 * n * WIDTH as u128 * v * (epochs + 2)
        + batches as u128 * 128 * d * (1 + (b * WIDTH as u128).ilog2() as u128);
    let total = init + train;
    let buffer = 256 * d + 2048 * n + 256 * input.route.len() as u128;
    if total > WORK_LIMIT || buffer > 56 * 1024u128.pow(3) {
        return Err(
            format!("shared-head forecast refused: {total} kernel work, {buffer} bytes").into(),
        );
    }
    Ok(Bounds {
        coefficients: d.try_into()?,
        gradient_batches: batches,
        initialization_work: init.try_into()?,
        training_work: train.try_into()?,
        total_kernel_work: total.try_into()?,
        buffer_bytes: buffer.try_into()?,
        prediction_work: (128 * WIDTH as u128 * v).try_into()?,
    })
}

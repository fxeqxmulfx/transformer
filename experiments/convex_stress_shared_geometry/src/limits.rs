//! Prospective scalar-work and storage allowances, including compact execution,
//! implied dense heads, every candidate, exclusion fold, local control and replay.
use crate::*;
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Bounds {
    pub maximum_stored_parameters: usize,
    pub implied_complete_coefficients: usize,
    pub candidate_fits: usize,
    pub candidate_scores: usize,
    pub candidate_scored_records: usize,
    pub gradient_batches: usize,
    pub constructor_work: u64,
    pub training_work: u64,
    pub independent_product_work: u64,
    pub one_run_kernel_work: u64,
    pub run_and_replay_kernel_work: u64,
    pub buffer_bytes: u64,
    pub prediction_work: u64,
}
pub fn forecast(input: &Input, policy: &Policy) -> Result<Bounds> {
    input.validate()?;
    policy.validate()?;
    let n = input.records.len() as u128;
    let b = input.blocks as u128;
    let v = input.vocabulary as u128;
    let (m, d) = (128u128, 68u128);
    let q = 2 * m + b;
    let expanded = b * (1 + 2 * m) * v;
    let candidates = 3 * (2 * 3 + 1) * 16 * 4;
    let fits = 3 * 3 * (2 * 3 + 1) * 16 + 1;
    let batches = match policy.mode {
        Mode::Full => policy.iterations,
        Mode::Mini => policy.iterations * input.records.len().div_ceil(policy.batch),
        Mode::Forward => 0,
    };
    if batches > 4096 {
        return Err("shared geometry gradient allowance exceeded".into());
    }
    let local = convex_stress_shared_head::limits::forecast(input, &Policy::forward())?;
    let construction = 4 * 3 * (64 * n * d * d + 32 * d * d * d + 64 * 16 * b * d)
        + 64 * fits * n * d
        + 64 * candidates * n * (m + v)
        + 256 * (3 * candidates + 1) * (n + b + q + input.route.len() as u128)
        + 3 * local.total_kernel_work as u128;
    let epochs = policy.iterations as u128;
    let train = 64 * n * (m + v) * (2 * epochs + 3)
        + batches as u128 * (128 * q * (1 + q.ilog2() as u128) + 64 * b * m);
    // One complete raw-class product check for each candidate and every original
    // training record of the final model. Feature generation includes the DFT.
    let products = 64 * (3 * candidates + n + 1) * (m * m + m * v);
    let total = construction + train + products;
    let buffers = 64 * expanded
        + 256 * fits * (b + q)
        + 256 * n * m
        + 256 * input.route.len() as u128
        + 3 * local.buffer_bytes as u128;
    if 2 * total > 1_000_000_000_000 || buffers > 56 * 1024u128.pow(3) {
        return Err(format!(
            "shared geometry forecast refused: {} work, {buffers} bytes",
            2 * total
        )
        .into());
    }
    Ok(Bounds {
        maximum_stored_parameters: q.try_into()?,
        implied_complete_coefficients: expanded.try_into()?,
        candidate_fits: fits.try_into()?,
        candidate_scores: (3 * candidates).try_into()?,
        candidate_scored_records: (candidates * n).try_into()?,
        gradient_batches: batches,
        constructor_work: construction.try_into()?,
        training_work: train.try_into()?,
        independent_product_work: products.try_into()?,
        one_run_kernel_work: total.try_into()?,
        run_and_replay_kernel_work: (2 * total).try_into()?,
        buffer_bytes: buffers.try_into()?,
        prediction_work: (64 * (m * m + m * v)).try_into()?,
    })
}

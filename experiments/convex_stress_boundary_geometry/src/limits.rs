//! Prospective scalar work and buffers for every constructor, fold, complete
//! tensor, final update, independent product and exact internal reconstruction.
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
    let (n, b, v) = (
        input.records.len() as u128,
        input.blocks as u128,
        input.vocabulary as u128,
    );
    let (m, d, h, q) = (8u128, 15u128, 9u128, 25u128);
    let route = input.route.len() as u128;
    let candidates = fit::configuration_count() as u128;
    let fits = 3 * candidates / (GAINS.len() * COMMONS.len()) as u128 + 1;
    let batches = match policy.mode {
        Mode::Full => policy.iterations,
        Mode::Mini => policy.iterations * input.records.len().div_ceil(policy.batch),
        Mode::Forward => 0,
    };
    if batches > 4096 {
        return Err("gradient allowance exceeded".into());
    }
    let local = convex_stress_shared_head::limits::forecast(input, &Policy::forward())?;
    let constructor = 64 * 16 * (route * h + n * d * d + b * d * d)
        + 64 * fits * (d * d * d + n * d)
        + 64 * candidates * n * (m + v + h)
        + 64 * (3 * candidates + 1) * (route + n + q + b * (m * h * 4 + v + h * 4))
        + 3 * local.total_kernel_work as u128;
    let train = 64 * n * (m + v + h) * (2 * policy.iterations as u128 + 3)
        + batches as u128
            * 64
            * (q * (1 + q.ilog2() as u128) + b * (m * h * 4 + v + h * 4) + route + n);
    let products = 64 * (3 * candidates + n + 1) * (m * m + m * v * h);
    let total = constructor + train + products;
    let expanded = b * (1 + 2 * m) * v;
    let buffers = 64 * expanded
        + 256 * fits * (b * (h + 16) + q)
        + 256 * n * (m + d)
        + 256 * route
        + 3 * local.buffer_bytes as u128;
    if 2 * total > 2_000_000_000_000 || buffers > 56 * 1024u128.pow(3) {
        return Err(format!(
            "boundary forecast refused: {} work, {buffers} bytes",
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
        constructor_work: constructor.try_into()?,
        training_work: train.try_into()?,
        independent_product_work: products.try_into()?,
        one_run_kernel_work: total.try_into()?,
        run_and_replay_kernel_work: (2 * total).try_into()?,
        buffer_bytes: buffers.try_into()?,
        prediction_work: (64 * (m * m + m * v * h)).try_into()?,
    })
}

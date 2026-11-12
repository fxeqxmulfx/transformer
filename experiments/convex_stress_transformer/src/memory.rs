//! Conservative live-buffer forecast for the serial streamed native engine.
//! Includes capacity growth, Adam moments, transposed weights, gradients,
//! retained activations and Kernite packing. Dataset storage is reported separately.
use crate::{Result, layout::Dims};

pub const BUFFER_LIMIT: usize = 2 * 1024 * 1024 * 1024;
pub const MAX_MICROBATCH: usize = 32;

pub fn validate(d: Dims) -> Result<()> {
    if !(2..=4096).contains(&d.vocab)
        || !(2..=1024).contains(&d.d)
        || d.heads == 0
        || !d.d.is_multiple_of(d.heads)
        || !(1..=32).contains(&d.layers)
        || d.dff != 4 * d.d
        || !(1..=8192).contains(&d.ctx)
    {
        return Err("unsupported transformer dimensions".into());
    }
    Ok(())
}

pub fn bytes(d: Dims, microbatch: usize) -> Result<usize> {
    validate(d)?;
    if !(1..=MAX_MICROBATCH).contains(&microbatch) {
        return Err("microbatch must be between one and 32".into());
    }
    // All arithmetic below uses u128, before any vector or model allocation.
    let (b, t, w, f, h, v, l, p) = (
        microbatch as u128,
        d.ctx as u128,
        d.d as u128,
        d.dff as u128,
        d.heads as u128,
        d.vocab as u128,
        d.layers as u128,
        d.len() as u128,
    );
    let rows = b * t;
    let traces = l * (rows * (9 * w + f + 2) + b * h * t * t);
    let temporary = rows * (7 * w + 2 * f + v);
    let parameters_and_positions = 10 * p + t * w;
    // Packing may use parameter rows as well as activation rows.
    let packed = rows.max(f).max(v).div_ceil(6) * 6 * 768;
    let bound =
        8 * (traces + temporary + parameters_and_positions + packed) + 32 * rows + 4 * 1024 * 1024;
    usize::try_from(bound).map_err(|_| "transformer allocation forecast overflow".into())
}

pub fn check(d: Dims, microbatch: usize) -> Result<usize> {
    let bytes = bytes(d, microbatch)?;
    if bytes > BUFFER_LIMIT {
        return Err(format!("transformer buffer forecast {bytes} exceeds {BUFFER_LIMIT}").into());
    }
    Ok(bytes)
}

/// Largest admitted chunk up to the requested batch; no task or label inputs.
pub fn microbatch(d: Dims, batch: usize) -> Result<usize> {
    if batch == 0 {
        return Err("empty optimizer batch".into());
    }
    (1..=batch.min(MAX_MICROBATCH))
        .rev()
        .find(|&b| check(d, b).is_ok())
        .ok_or_else(|| "even one transformer example exceeds the buffer limit".into())
}

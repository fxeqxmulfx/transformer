//! Nested limits exclude caller-owned original inputs and metadata. The
//! outer admission::Budget counts those before these operations are called.
use crate::Result;
pub fn remaining(limit: u64, live: u128) -> Result<u64> {
    if limit == 0 || limit > joint::admission::CORE_BYTES || live >= limit as u128 {
        return Err("complete comparison buffers exceed local admission".into());
    }
    Ok(limit - live as u64)
}
pub fn vector<T>(count: usize, limit: u64) -> Result<Vec<T>> {
    remaining(
        limit,
        std::mem::size_of::<Vec<T>>() as u128 + count as u128 * std::mem::size_of::<T>() as u128,
    )?;
    let mut out = Vec::new();
    out.try_reserve_exact(count)?;
    if out.capacity() != count {
        return Err("comparison vector capacity differs from admission".into());
    }
    Ok(out)
}

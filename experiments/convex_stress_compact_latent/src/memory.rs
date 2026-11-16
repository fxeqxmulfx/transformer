//! Local capacity checks. Callers subtract every simultaneously live object
//! before constructing another one. Original-input process admission is separate.
use crate::Result;
use std::mem::size_of;

pub(crate) fn remaining(limit: u64, occupied: u128) -> Result<u64> {
    if limit == 0 || limit > 96_000_000_000 || occupied >= limit as u128 {
        return Err("complete simultaneous buffers exceed admitted storage".into());
    }
    Ok(limit - occupied as u64)
}
pub(crate) fn vector_bytes<T>(count: usize) -> u128 {
    size_of::<Vec<T>>() as u128 + count as u128 * size_of::<T>() as u128
}
pub(crate) fn vector<T>(count: usize, limit: u64) -> Result<Vec<T>> {
    remaining(limit, vector_bytes::<T>(count))?;
    let mut out = Vec::new();
    out.try_reserve_exact(count)?;
    if out.capacity() != count {
        return Err("allocator capacity differs from admitted vector size".into());
    }
    Ok(out)
}

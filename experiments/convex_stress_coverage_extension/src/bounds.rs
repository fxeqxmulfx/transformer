//! Both CM39 blocks are stored and charged. These are conservative allocation
//! and scalar-work forecasts, distinct from measured RSS and exact Lean bounds.
use crate::*;
pub const BUFFER_CAP: u64 = 56 * 1024 * 1024 * 1024;
pub const WORK_CAP: u64 = 32_000_000_000;
pub const RECORD_CAP: usize = 200_000;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bounds {
    pub real_coefficients: usize,
    pub stored_float_scalars: usize,
    pub radius: f64,
    pub buffer_bytes: u64,
    pub head_audit_work: u64,
}

pub fn combine(
    prior: &kernel::limits::Bounds,
    supplement: &kernel::limits::Bounds,
    certificate_bytes: u64,
    records: usize,
) -> Result<Bounds> {
    if records == 0 || records > RECORD_CAP {
        return Err("original records exceed the200000-record limit".into());
    }
    let add = |a: u64, b: u64| a.checked_add(b).ok_or("combined bound overflow");
    let real_coefficients = prior
        .real_coefficients
        .checked_add(supplement.real_coefficients)
        .ok_or("coefficient count overflow")?;
    let phase_entries = prior
        .phase_entries
        .checked_add(supplement.phase_entries)
        .ok_or("phase count overflow")?;
    let stored_float_scalars = real_coefficients
        .checked_add(real_coefficients / 2)
        .and_then(|x| phase_entries.checked_mul(2).and_then(|p| x.checked_add(p)))
        .ok_or("stored scalar count overflow")?;
    let buffer_bytes = add(
        add(
            add(prior.buffer_bytes, supplement.buffer_bytes)?,
            certificate_bytes,
        )?,
        128 * records as u64,
    )?;
    let head_audit_work = add(prior.audit_work, supplement.audit_work)?;
    let states = prior
        .states
        .checked_add(supplement.states)
        .ok_or("state count overflow")?;
    let radius = 64. * states as f64;
    if buffer_bytes > BUFFER_CAP || head_audit_work > WORK_CAP || !radius.is_finite() {
        return Err("combined extension exceeds memory or head-audit forecast".into());
    }
    Ok(Bounds {
        real_coefficients,
        stored_float_scalars,
        radius,
        buffer_bytes,
        head_audit_work,
    })
}

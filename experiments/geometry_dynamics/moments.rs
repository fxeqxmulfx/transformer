//! Exact-real .9/.95 invariant, checked on saved FP32 buffers with fixed slack.
use crate::{Result, evidence::require};
use serde_json::{Value, json};
pub const C: f64 = 19. / 14.;
pub fn direction_bound(step: usize) -> f64 {
    if step == 0 {
        0.
    } else {
        C.sqrt() * (1. - 0.95f64.powi(step as i32)).sqrt() / (1. - 0.9f64.powi(step as i32))
    }
}
pub fn check(bytes: &[u8], count: usize, step: usize) -> Result<Value> {
    require(bytes.len() == 8 + 12 * count, "moment state shape changed")?;
    let get = |at| f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as f64;
    let mut excess = 0f64;
    let mut maximum = 0f64;
    for k in 0..count {
        let m = get(8 + 4 * (count + k));
        let v = get(8 + 4 * (2 * count + k));
        require(
            m.is_finite() && v.is_finite() && v >= 0.,
            "invalid saved moment",
        )?;
        excess = excess.max((m * m - C * v) / (m * m).max(C * v).max(1e-30));
        if step == 0 {
            require(m == 0. && v == 0., "initial moments are not zero")?;
        } else {
            let actual = (m / (1. - 0.9f64.powi(step as i32))).abs()
                / (v.sqrt() / (1. - 0.95f64.powi(step as i32)).sqrt() + 1e-8);
            maximum = maximum.max(actual);
        }
    }
    require(
        excess <= 1e-6,
        "saved moment invariant exceeds fixed allowance",
    )?;
    require(
        maximum <= direction_bound(step) * (1. + 1e-6),
        "adaptive direction exceeds invariant bound",
    )?;
    Ok(
        json!({"coordinates":count,"max_relative_invariant_excess":excess,
        "max_adaptive_direction":maximum,"analytic_direction_bound":direction_bound(step)}),
    )
}

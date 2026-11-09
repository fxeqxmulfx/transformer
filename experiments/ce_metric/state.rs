//! Checkpoint decoding, support and schedule ported from future_forcing at 72f30df3.
use crate::{
    Result,
    evidence::{require, word},
};
use serde_json::Value;
pub const E: usize = 97 * 128;
pub struct State {
    pub p: Vec<f64>,
    pub m: Vec<f64>,
    pub v: Vec<f64>,
}
pub fn decode(bytes: &[u8], step: usize) -> Result<State> {
    let count = E + 3 * 128 * 682;
    require(
        bytes.len() == 8 + 12 * count && word(bytes, 0) == step as u64,
        "checkpoint shape/time",
    )?;
    let get = |block| {
        (0..E)
            .map(|k| {
                let at = 8 + 4 * (block * count + k);
                f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as f64
            })
            .collect::<Vec<_>>()
    };
    let s = State {
        p: get(0),
        m: get(1),
        v: get(2),
    };
    require(
        s.p.iter().chain(&s.m).chain(&s.v).all(|v| v.is_finite()) && s.v.iter().all(|&v| v >= 0.),
        "nonfinite/negative checkpoint",
    )?;
    Ok(s)
}
pub fn support(queries: &Value) -> Result<Vec<bool>> {
    let mut a = vec![false; 97];
    let mut n = 0;
    for q in queries.as_array().ok_or("training support")? {
        if q[2] == 0 {
            let t = q[1].as_u64().ok_or("training label")? as usize;
            require(t < 97, "label range")?;
            a[t] = true;
            n += 1;
        }
    }
    require(
        n == 4704 && a.iter().filter(|&&a| a).count() == 33,
        "training inventory",
    )?;
    Ok(a)
}
pub fn rate(cohort: &str, t: usize) -> f64 {
    let peak = if cohort == "lr001" { 0.001 } else { 0.01 };
    let eta = if t < 200 {
        peak * t as f64 / 200.
    } else if cohort == "settle" {
        if t <= 1000 {
            peak
        } else if t < 4000 {
            let u = (t - 1000) as f64 / 3000.;
            peak + (1e-4 - peak) * 0.5 * (1. - (std::f64::consts::PI * u).cos())
        } else if t <= 8000 {
            1e-4
        } else {
            let u = (t - 8000) as f64 / 1999.;
            1e-4 + (1e-8 - 1e-4) * 0.5 * (1. - (std::f64::consts::PI * u).cos())
        }
    } else if t < 8000 {
        peak
    } else {
        let u = (t - 8000) as f64 / 1999.;
        peak + (1e-8 - peak) * 0.5 * (1. - (std::f64::consts::PI * u).cos())
    };
    (eta as f32) as f64
}
pub fn metric(s: &State, active: &[bool], t: usize) -> (f64, Vec<f64>) {
    let b2 = (1. - 0.95f64.powi(t as i32)).sqrt();
    let d: Vec<_> =
        s.v.iter()
            .map(|v| {
                if t == 0 {
                    1e8
                } else {
                    1. / (v.sqrt() / b2 + 1e-8)
                }
            })
            .collect();
    let alpha = d.iter().copied().fold(f64::INFINITY, f64::min);
    let r = (0..128)
        .map(|k| {
            (0..97)
                .filter(|&c| !active[c])
                .map(|c| d[c * 128 + k])
                .fold(f64::INFINITY, f64::min)
                / alpha
        })
        .collect();
    (alpha, r)
}

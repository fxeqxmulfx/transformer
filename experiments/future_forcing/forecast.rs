//! FF1-FF3: constructor receives one checkpoint, never a later update.
use crate::{Result, classes::Partition, evidence::require};
use serde_json::{Value, json};
pub const E: usize = 97 * 128;
pub const C: f64 = 19. / 14.;
pub const Q: f64 = 81. / 95.;
pub const RELATIVE: f64 = 1e-6;
pub const THRESHOLDS: [f64; 4] = [0., 0.5, 0.6, 2. / 3.];
pub const NAMES: [&str; 2] = ["warm_state", "uniform"];
pub fn norm(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>().sqrt()
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
pub struct State {
    pub p: Vec<f64>,
    pub m: Vec<f64>,
    pub v: Vec<f64>,
}
pub fn state(bytes: &[u8], step: usize) -> Result<State> {
    let count = 12416 + 3 * 128 * 682;
    require(
        bytes.len() == 8 + 12 * count && crate::evidence::word(bytes, 0) == step as u64,
        "invalid checkpoint shape/time",
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
pub struct Anchor {
    pub state: State,
    pub active: Vec<bool>,
    pub initial: [f64; 2],
    pub sum_ratio: [f64; 2],
    pub max_ratio: f64,
}
impl Anchor {
    pub fn new(state: State, active: Vec<bool>) -> Result<Self> {
        require(
            active.len() == 97 && active.iter().filter(|&&a| a).count() == 33,
            "support changed",
        )?;
        let part = Partition::new(active.clone(), 128).project(&state.p);
        let initial = [norm(&part[0]), norm(&part[1]).hypot(norm(&part[2]))];
        let mut sum_ratio = [0.; 2];
        let mut max_ratio = 0f64;
        for k in 0..E {
            require(
                state.v[k] > 0. || state.m[k] == 0.,
                "zero variance with nonzero moment",
            )?;
            let r = if state.v[k] == 0. {
                0.
            } else {
                (state.m[k].powi(2) / state.v[k]).next_up()
            };
            sum_ratio[0] += r;
            if active[k / 128] {
                sum_ratio[1] += r;
            }
            max_ratio = max_ratio.max(r);
        }
        require(
            max_ratio <= C * (1. + RELATIVE),
            "anchor cone outside existing allowance",
        )?;
        Ok(Self {
            state,
            active,
            initial,
            sum_ratio,
            max_ratio,
        })
    }
    pub fn metadata(&self) -> Value {
        json!({"active":self.active,"initial_norms":self.initial,"sum_initial_ratios":self.sum_ratio,
            "max_initial_ratio":self.max_ratio,"coordinates":E,"active_coordinates":33*128})
    }
}
/// Each track: total/active direction norms, max direction, rounding cap,
/// coordinate parameter bound, active/total tube radii, share lower bound.
pub type Track = [f64; 8];
pub struct Forecast {
    pub anchor: usize,
    pub cohort: String,
    pub q: f64,
    pub tracks: [Track; 2],
}
impl Forecast {
    pub fn new(a: &Anchor, anchor: usize, cohort: &str) -> Self {
        let maximum = a.state.p.iter().map(|x| x.abs()).fold(0., f64::max);
        let mut tracks = [[0.; 8]; 2];
        for t in &mut tracks {
            t[4] = maximum;
        }
        Self {
            anchor,
            cohort: cohort.into(),
            q: 1.,
            tracks,
        }
    }
    pub fn step(&mut self, a: &Anchor, h: usize) -> Value {
        let step = self.anchor + h;
        let eta = rate(&self.cohort, step - 1);
        let decay = 1. - 0.1 * eta;
        let bias = (1. - 0.95f64.powi(step as i32)).sqrt() / (1. - 0.9f64.powi(step as i32));
        let memory = Q.powi(h as i32);
        self.q *= decay;
        for (j, t) in self.tracks.iter_mut().enumerate() {
            let cone = |sum: f64, dim: f64| {
                if j == 0 {
                    dim * C + (sum - dim * C) * memory
                } else {
                    dim * C
                }
            };
            t[0] = cone(a.sum_ratio[0], E as f64).max(0.).sqrt() * bias;
            t[1] = cone(a.sum_ratio[1], (33 * 128) as f64).max(0.).sqrt() * bias;
            t[2] = cone(a.max_ratio, 1.).max(0.).sqrt() * bias;
            t[3] = 8. * f32::EPSILON as f64 * (1. + t[4] + eta * t[2] * (1. + RELATIVE));
            t[4] = decay * t[4] + eta * t[2] * (1. + RELATIVE) + t[3];
            t[5] = decay * t[5] + eta * t[1] * (1. + RELATIVE) + (33f64 * 128.).sqrt() * t[3];
            t[6] = decay * t[6] + eta * t[0] * (1. + RELATIVE) + (E as f64).sqrt() * t[3];
            let lower = (self.q * a.initial[0] - t[5]).max(0.);
            let upper = self.q * a.initial[1] + t[6];
            t[7] = if lower == 0. {
                0.
            } else {
                lower * lower / (lower * lower + upper * upper)
            };
        }
        json!([step, eta, self.q, self.tracks])
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sharp_step_and_warm_cone_cover_nonconstant_bounded_gradients() {
        let (m, v) = (0.1f64, 0.01f64);
        let g = 19. * v / (9. * m);
        let next_m = 0.9 * m + 0.1 * g;
        let next_v = 0.95 * v + 0.05 * g * g;
        assert!(g.abs() < 1.);
        assert!((next_m * next_m / next_v - (Q * m * m / v + 0.2)).abs() < 1e-14);
        for initial in [(0., 0.), (m, v), (-m, v), (0., v)] {
            let (mut m, mut v) = initial;
            let r = if v == 0. { 0. } else { m * m / v };
            for h in 1..=300 {
                let g = ((h * 37 % 101) as f64 - 50.) / 50.;
                m = 0.9 * m + 0.1 * g;
                v = 0.95 * v + 0.05 * g * g;
                let k = C + (r - C) * Q.powi(h);
                assert!(m * m <= k * v + 1e-14);
            }
        }
    }
    #[test]
    fn schedules_preserve_source_boundaries_and_nonzero_hold() {
        assert_eq!(rate("settle", 0), 0.);
        assert_eq!(rate("settle", 200), 0.01f32 as f64);
        assert_eq!(rate("settle", 4000), 1e-4f32 as f64);
        assert_eq!(rate("settle", 8000), 1e-4f32 as f64);
        for name in ["lr001", "lr01", "settle"] {
            assert_eq!(rate(name, 9999), 1e-8f32 as f64);
            assert!(rate(name, 9000) > rate(name, 9500));
        }
        assert_eq!(rate("lr001", 7999), 0.001f32 as f64);
        assert_eq!(rate("lr01", 7999), 0.01f32 as f64);
    }
}

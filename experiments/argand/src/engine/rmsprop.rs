//! Bias-corrected uncentered RMSProp with decoupled weight decay. Only the
//! mean square of each globally clipped gradient coordinate is retained.
//! The Adam variants retain two moving averages. Normalized AdamW starts
//! both gains at one and updates the two shared scalars by recursion.
use super::{Engine, F, Result, adamw::RatePlan, grid, page::Observations};
use crate::schedule::Schedule;
use serde_json::{Value, json};

#[derive(Clone, Copy)]
pub(crate) enum MomentRates {
    Equal,
    HalfSecondGain,
    Normalized { first_gain: f64, second_gain: f64 },
}

impl MomentRates {
    pub const NORMALIZED: Self = Self::Normalized {
        first_gain: 1.0,
        second_gain: 1.0,
    };

    pub fn label(self) -> &'static str {
        match self {
            Self::Equal => "equal-moment-wsd",
            Self::HalfSecondGain => "uncorrected-adam-wsd",
            Self::Normalized { .. } => "normalized-adamw-wsd",
        }
    }

    /// Coefficients of the old moment and new gradient, rounded separately.
    fn next_gains(&mut self) -> [f32; 4] {
        match self {
            Self::Equal => [0.9, 0.1, 0.9, 0.1],
            Self::HalfSecondGain => [0.9, 0.1, 0.95, 0.05],
            Self::Normalized {
                first_gain,
                second_gain,
            } => {
                let (a, b) = (*first_gain, *second_gain);
                *first_gain = a / (0.9 + a);
                *second_gain = b / (0.95 + b);
                [(1.0 - a) as f32, a as f32, (1.0 - b) as f32, b as f32]
            }
        }
    }
}

pub(crate) struct RmsProp {
    mean_square: F,
    first: Option<(F, MomentRates)>,
    schedule: Schedule,
    scale: f64,
    decay: f64,
    completed: usize,
}

pub(crate) fn settings(scale: f64, decay: f64, schedule: Schedule) -> Value {
    json!({"peak":crate::schedule::PEAK*scale,"min_rate":1e-8*scale,"budget":schedule.budget,
        "warmup":schedule.warmup,"cooldown_start":schedule.cooldown_start,"weight_decay":decay,
        "beta2":0.95,"epsilon":1e-8,"clip_norm":1,"bias_correction":true,"centered":false})
}

pub(crate) fn moment_settings(
    scale: f64,
    decay: f64,
    schedule: Schedule,
    rates: MomentRates,
) -> Value {
    let normalized = matches!(rates, MomentRates::Normalized { .. });
    let mut settings = json!({"peak":crate::schedule::PEAK*scale,"min_rate":1e-8*scale,"budget":schedule.budget,
        "warmup":schedule.warmup,"cooldown_start":schedule.cooldown_start,"weight_decay":decay,
        "beta1":0.9,"beta2":if matches!(rates, MomentRates::Equal) {0.9} else {0.95},
        "epsilon":1e-8,"clip_norm":1,"bias_correction":normalized});
    if normalized {
        settings["normalization"] = "recursive-gains".into();
    }
    settings
}

impl RmsProp {
    pub fn new(engine: &Engine, schedule: Schedule, scale: f64, decay: f64) -> Result<Self> {
        Ok(Self {
            mean_square: engine.cuda.upload(&vec![0.0; engine.parameters.len()])?,
            first: None,
            schedule,
            scale,
            decay,
            completed: 0,
        })
    }

    pub fn with_first_moment(mut self, engine: &Engine, rates: MomentRates) -> Result<Self> {
        self.first = Some((
            engine.cuda.upload(&vec![0.0; engine.parameters.len()])?,
            rates,
        ));
        Ok(self)
    }

    pub fn step(&mut self, engine: &mut Engine, batch: Observations<'_>) -> Result<()> {
        batch.forward(engine)?;
        engine.backward()?;
        self.apply(engine)
    }

    fn apply(&mut self, engine: &Engine) -> Result<()> {
        let p = &engine.parameters;
        let norm = p.reduce_norm(&engine.optimizer, &p.grad)?;
        let rate = RatePlan::Scaled(self.scale).rate(self.schedule, self.completed);
        let shrink = 1.0 - rate * self.decay as f32;
        // SAFETY: one thread per coordinate in equally sized weights, gradients and EMA buffers.
        unsafe {
            if let Some((first, rates)) = &mut self.first {
                let [beta1, gain1, beta2, gain2] = rates.next_gains();
                engine.kernel.moving_moments_decay(
                    grid(p.len()),
                    p.value.ptr(),
                    first.ptr(),
                    self.mean_square.ptr(),
                    p.grad.ptr(),
                    norm.ptr(),
                    p.len() as u32,
                    rate,
                    shrink,
                    beta1,
                    gain1,
                    beta2,
                    gain2,
                )?;
            } else {
                let bias = (1.0 - 0.95_f64.powf((self.completed + 1) as f64)).sqrt() as f32;
                engine.kernel.rmsprop_decay(
                    grid(p.len()),
                    p.value.ptr(),
                    self.mean_square.ptr(),
                    p.grad.ptr(),
                    norm.ptr(),
                    p.len() as u32,
                    rate,
                    shrink,
                    bias,
                )?;
            }
        }
        self.completed += 1;
        Ok(())
    }

    pub fn metrics(&self) -> Value {
        json!({"step":self.completed,"rate":self.completed.checked_sub(1).map(|t|
            RatePlan::Scaled(self.scale).rate(self.schedule,t))})
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recursive_gains_match_normalized_exponential_weights_through_200k() {
        let mut rates = MomentRates::NORMALIZED;
        for update in 1..=200_000 {
            let gains = rates.next_gains();
            for (decay, pair) in [(0.9_f64, &gains[..2]), (0.95_f64, &gains[2..])] {
                let new = (1.0 - decay) / (1.0 - decay.powi(update));
                assert_eq!(pair[0], (1.0 - new) as f32);
                assert_eq!(pair[1], new as f32);
            }
        }
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn normalized_buffers_and_weights_match_the_bias_corrected_reference() -> Result<()> {
        let shape = crate::suite::Shape {
            width: 16,
            heads: 2,
            layers: 1,
            vocab: 4,
            max_batch: 2,
            max_length: 2,
        };
        let mut engine = Engine::new(shape, 1, 8)?;
        let schedule = super::super::adamw::wsd(60);
        let mut opt = RmsProp::new(&engine, schedule, 3.0, 0.1)?
            .with_first_moment(&engine, MomentRates::NORMALIZED)?;
        let mut grad = vec![0.0; engine.parameters.len()];
        let mut weights = engine.weights()?;
        let (mut raw_m, mut raw_v) = ([0.0_f64; 2], [0.0_f64; 2]);
        let pairs = [[2e-9_f32, -1e-8], [3.0, 4.0], [0.0, 0.0], [-0.6, 0.3]];
        for t in 0..60 {
            let pair = pairs[t % pairs.len()];
            grad[..2].copy_from_slice(&pair);
            engine.parameters.grad.write(&grad)?;
            opt.apply(&engine)?;
            let first = opt.first.as_ref().unwrap().0.read()?;
            let second = opt.mean_square.read()?;
            let actual = engine.weights()?;
            let norm = f64::from(pair[0]).hypot(f64::from(pair[1])).max(1.0);
            let rate = f64::from(RatePlan::Scaled(3.0).rate(schedule, t));
            let update = (t + 1) as i32;
            for i in 0..2 {
                let g = f64::from(pair[i]) / norm;
                raw_m[i] = 0.9 * raw_m[i] + 0.1 * g;
                raw_v[i] = 0.95 * raw_v[i] + 0.05 * g * g;
                let m = raw_m[i] / (1.0 - 0.9_f64.powi(update));
                let v = raw_v[i] / (1.0 - 0.95_f64.powi(update));
                weights[i] = ((1.0 - rate * 0.1) * f64::from(weights[i])
                    - rate * m / (v.sqrt() + 1e-8)) as f32;
                assert!((f64::from(first[i]) - m).abs() < 1e-7);
                assert!((f64::from(second[i]) - v).abs() < 1e-7);
                assert!((actual[i] - weights[i]).abs() < 4e-7);
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn matches_clipped_bias_corrected_rms_with_only_a_second_moment() -> Result<()> {
        let shape = crate::suite::Shape {
            width: 16,
            heads: 2,
            layers: 1,
            vocab: 4,
            max_batch: 2,
            max_length: 2,
        };
        let mut engine = Engine::new(shape, 1, 13)?;
        let schedule = super::super::adamw::wsd(20);
        let mut opt = RmsProp::new(&engine, schedule, 1.0, 0.2)?;
        let mut w = engine.weights()?;
        let mut v = [0.0_f32; 2];
        for (t, raw) in [[3.0_f32, 4.0], [0.0, 0.0], [-0.3, 0.4], [1e-7, 0.0]]
            .into_iter()
            .enumerate()
        {
            let mut grad = vec![0.0; w.len()];
            grad[..2].copy_from_slice(&raw);
            engine.parameters.grad.write(&grad)?;
            opt.apply(&engine)?;
            let norm = (raw[0] * raw[0] + raw[1] * raw[1]).sqrt().max(1.0);
            let rate = RatePlan::Wsd.rate(schedule, t);
            let bias = (1.0 - 0.95_f64.powi((t + 1) as i32)).sqrt() as f32;
            let actual = engine.weights()?;
            let moment = opt.mean_square.read()?;
            for i in 0..2 {
                let g = raw[i] / norm;
                v[i] = 0.95 * v[i] + 0.05 * g * g;
                w[i] = w[i] * (1.0 - rate * 0.2) - rate * g / (v[i].sqrt() / bias + 1e-8);
                assert!((moment[i] - v[i]).abs() < 1e-7);
                assert!((actual[i] - w[i]).abs() < 2e-7);
            }
            w[2] *= 1.0 - rate * 0.2;
            assert_eq!(actual[2], w[2]);
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn uncorrected_moment_updates_match_scalar_recurrences() -> Result<()> {
        check_uncorrected(MomentRates::Equal, 0.9, 0.1, 1.0)?;
        check_uncorrected(MomentRates::HalfSecondGain, 0.95, 0.05, 19.0 / 14.0)
    }

    fn check_uncorrected(rates: MomentRates, beta2: f32, gain: f32, bound: f32) -> Result<()> {
        let shape = crate::suite::Shape {
            width: 16,
            heads: 2,
            layers: 1,
            vocab: 4,
            max_batch: 2,
            max_length: 2,
        };
        let mut engine = Engine::new(shape, 1, 8)?;
        let schedule = super::super::adamw::wsd(20);
        let mut opt =
            RmsProp::new(&engine, schedule, 2.0, 0.1)?.with_first_moment(&engine, rates)?;
        let mut w = engine.weights()?[0];
        let (mut m, mut v) = (0.0_f32, 0.0_f32);
        for (t, pair) in [
            [0.0_f32, 0.0],
            [3.0, 4.0],
            [0.0, 0.0],
            [-0.6, 0.0],
            [1e-9, 0.0],
        ]
        .into_iter()
        .enumerate()
        {
            let mut grad = vec![0.0; engine.parameters.len()];
            grad[..2].copy_from_slice(&pair);
            engine.parameters.grad.write(&grad)?;
            opt.apply(&engine)?;
            let g = pair[0] / pair[0].hypot(pair[1]).max(1.0);
            m = 0.9 * m + 0.1 * g;
            v = beta2 * v + gain * (g * g);
            let d = m / (v.sqrt() + 1e-8);
            assert!(d * d <= bound + 1e-6);
            let rate = RatePlan::Scaled(2.0).rate(schedule, t);
            w = (1.0 - rate * 0.1) * w - rate * d;
            let actual_m = opt.first.as_ref().unwrap().0.read()?[0];
            let actual_v = opt.mean_square.read()?[0];
            assert!((actual_m - m).abs() < 1e-7 && (actual_v - v).abs() < 1e-7);
            assert!(actual_m * actual_m <= bound * actual_v + 1e-8);
            assert!((engine.weights()?[0] - w).abs() < 2e-7);
        }
        Ok(())
    }
}

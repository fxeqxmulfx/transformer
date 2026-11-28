//! The clipped, bias-corrected AdamW update of Argand at 7399c3bb.
use super::{Engine, F, Result, grid, page::Observations};
use crate::{random::Random, schedule::Schedule};
use kernite_cuda::Buffer;
use serde_json::{Value, json};

pub(crate) const DECAY: f32 = 0.1;
pub(crate) const FIRST_NOISE_STDDEV: f64 = 1e-6;

pub(crate) fn wsd(budget: usize) -> Schedule {
    Schedule {
        budget,
        warmup: budget / 50,
        cooldown_start: (budget as u128 * 80 / 100) as usize,
    }
}

#[derive(Clone, Copy)]
pub(crate) enum RatePlan {
    Wsd,
    Geometric {
        budget: usize,
    },
    Restart {
        stock_runs: usize,
        trial: usize,
        first_budget: usize,
    },
}

impl RatePlan {
    pub fn settings(self, schedule: Schedule) -> Value {
        let scale = self.scale(schedule.budget);
        json!({"weight_decay":0.1,"beta1":0.9,"beta2":0.95,"epsilon":1e-8,"clip_norm":1,
            "peak":crate::schedule::PEAK*scale,"min_rate":1e-8*scale,"budget":schedule.budget,
            "warmup":schedule.warmup,"cooldown_start":schedule.cooldown_start})
    }

    pub fn scale(self, budget: usize) -> f64 {
        match self {
            Self::Restart {
                stock_runs,
                trial,
                first_budget,
            } if trial >= stock_runs => {
                let horizon_ratio = (first_budget as f64 + 1.0) / (budget as f64 + 1.0);
                let trial_ratio = (stock_runs as f64 + 1.0) / (trial as f64 + 1.0);
                horizon_ratio.min(1.0) * trial_ratio
            }
            _ => 1.0,
        }
    }

    fn rate(self, schedule: Schedule, completed: usize) -> f32 {
        if let Self::Geometric { budget } = self
            && completed >= budget
        {
            return (1e-8 * 0.5_f64.powf((completed - budget) as f64)) as f32;
        }
        let t = completed.min(schedule.budget - 1);
        let base =
            if t >= schedule.warmup && t == schedule.cooldown_start && t + 1 == schedule.budget {
                // A one-point cooldown starts at its peak. It has no time interval to divide by.
                crate::schedule::PEAK as f32
            } else {
                schedule.rate(completed)
            };
        (f64::from(base) * self.scale(schedule.budget)) as f32
    }
}

pub(crate) struct AdamW {
    owners: Buffer<u32>,
    first: F,
    second: F,
    schedule: Schedule,
    plan: RatePlan,
    completed: usize,
    first_noise_seed: Option<u32>,
}

impl AdamW {
    pub fn new(engine: &Engine, schedule: Schedule, plan: RatePlan) -> Result<Self> {
        let n = u32::try_from(engine.parameters.len())?;
        if !n.is_multiple_of(2) {
            return Err(
                "the Argand parameter layout must contain complete coordinate pairs".into(),
            );
        }
        let owners: Vec<[u32; 5]> = (0..n).step_by(2).map(|i| [i, i + 1, 0, 0, 1]).collect();
        let zeros = vec![0.0; n as usize];
        Ok(Self {
            owners: engine.cuda.upload(owners.as_flattened())?,
            first: engine.cuda.upload(&zeros)?,
            second: engine.cuda.upload(&zeros)?,
            schedule,
            plan,
            completed: 0,
            first_noise_seed: None,
        })
    }

    /// Add independent Gaussian noise to the weights once, immediately
    /// after update one. The native start, both moment buffers and all
    /// scheduled rates are shared with ordinary AdamW.
    pub fn with_first_noise(mut self, seed: u32) -> Self {
        self.first_noise_seed = Some(seed);
        self
    }

    pub fn step(&mut self, engine: &mut Engine, batch: Observations<'_>) -> Result<()> {
        batch.forward(engine)?;
        engine.backward()?;
        self.apply(engine)
    }

    fn rate(&self, completed: usize) -> f32 {
        self.plan.rate(self.schedule, completed)
    }

    fn apply(&mut self, engine: &mut Engine) -> Result<()> {
        let p = &mut engine.parameters;
        let norm = p.reduce_norm(&engine.optimizer, &p.grad)?;
        let t = (self.completed + 1) as f64;
        // SAFETY: the identity ownership table covers every coordinate once;
        // all parameter and moment buffers have the same even length.
        unsafe {
            engine.optimizer.adam_coordinates(
                grid(p.len() / 2),
                p.value.ptr(),
                self.first.ptr(),
                self.second.ptr(),
                p.grad.ptr(),
                self.owners.ptr(),
                norm.ptr(),
                (p.len() / 2) as u32,
                self.rate(self.completed),
                DECAY,
                (1.0 - 0.9_f64.powf(t)) as f32,
                (1.0 - 0.95_f64.powf(t)).sqrt() as f32,
            )?;
        }
        self.completed += 1;
        if let Some(seed) = self.first_noise_seed.take() {
            let mut rng = Random::new(seed);
            let mut weights = p.value.read()?;
            for x in &mut weights {
                *x += rng.gauss(0.0, FIRST_NOISE_STDDEV) as f32;
            }
            p.value.write(&weights)?;
        }
        Ok(())
    }

    pub fn metrics(&self) -> Value {
        json!({"step":self.completed,"rate":self.completed.checked_sub(1).map(|t| self.rate(t))})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::suite::Shape;

    #[test]
    fn restart_scale_preserves_stock_trials_and_bounds_growing_horizons() {
        let plan = |trial| RatePlan::Restart {
            stock_runs: 12,
            trial,
            first_budget: 100_000,
        };
        assert_eq!(plan(11).scale(200_000), 1.0);
        assert_eq!(plan(12).scale(100_000), 1.0);
        assert_eq!(plan(13).scale(50_000), 13.0 / 14.0);
        assert_eq!(plan(25).scale(200_000), (100_001.0 / 200_001.0) * 0.5);
        let ordinary = Schedule {
            budget: 100_000,
            warmup: 2_000,
            cooldown_start: 80_000,
        };
        for t in 0..ordinary.budget {
            assert_eq!(plan(0).rate(ordinary, t), RatePlan::Wsd.rate(ordinary, t));
        }
        let short = Schedule {
            budget: 4,
            warmup: 0,
            cooldown_start: 3,
        };
        for t in 0..4 {
            assert_eq!(plan(25).rate(short, t), crate::schedule::PEAK as f32 / 2.0);
        }
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn first_noise_changes_only_weights_once_after_the_native_update() -> Result<()> {
        let shape = Shape {
            width: 16,
            heads: 2,
            layers: 1,
            vocab: 4,
            max_batch: 2,
            max_length: 3,
        };
        let mut plain = Engine::new(shape.clone(), 1, 5)?;
        let mut noisy = Engine::new(shape, 1, 5)?;
        let mut a = AdamW::new(&plain, wsd(8), RatePlan::Wsd)?;
        let mut b = AdamW::new(&noisy, wsd(8), RatePlan::Wsd)?.with_first_noise(73);
        assert_eq!(plain.weights()?, noisy.weights()?);
        let mut grad = vec![0.0; noisy.parameters.len()];
        grad[0] = 3.0;
        grad[1] = -4.0;
        for t in 0..3 {
            plain.parameters.grad.write(&grad)?;
            noisy.parameters.grad.write(&grad)?;
            a.apply(&mut plain)?;
            b.apply(&mut noisy)?;
            assert_eq!(a.first.read()?, b.first.read()?);
            assert_eq!(a.second.read()?, b.second.read()?);
            let mut expected = plain.weights()?;
            if t == 0 {
                let mut rng = Random::new(73);
                for x in &mut expected {
                    *x += rng.gauss(0.0, FIRST_NOISE_STDDEV) as f32;
                }
                assert_ne!(plain.weights()?, expected);
                plain.restore(&expected)?;
            }
            assert_eq!(noisy.weights()?, expected);
            grad.swap(0, 1);
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn clipped_moments_decay_and_wsd_match_scalar_coordinates() -> Result<()> {
        let shape = Shape {
            width: 16,
            heads: 2,
            layers: 1,
            vocab: 4,
            max_batch: 2,
            max_length: 3,
        };
        let mut e = Engine::new(shape, 1, 5)?;
        let wsd = Schedule {
            budget: 10,
            warmup: 2,
            cooldown_start: 8,
        };
        let mut opt = AdamW::new(&e, wsd, RatePlan::Geometric { budget: wsd.budget })?;
        let mut w = e.weights()?;
        let mut m = vec![0.0; w.len()];
        let mut v = m.clone();
        for t in 0..13 {
            let mut grad = vec![0.0; w.len()];
            let i = t % 3;
            grad[i] = if t % 2 == 0 { 4.0 } else { -0.4 };
            grad[i + 1] = if t % 2 == 0 { -3.0 } else { 0.2 };
            e.parameters.grad.write(&grad)?;
            opt.apply(&mut e)?;
            let norm = grad.iter().map(|g| g * g).sum::<f32>().sqrt().max(1.0);
            let rate = if t < 10 {
                wsd.rate(t)
            } else {
                1e-8 / (1 << (t - 10)) as f32
            };
            assert_eq!(opt.rate(t), rate);
            for j in 0..w.len() {
                let g = grad[j] / norm;
                m[j] = 0.9 * m[j] + 0.1 * g;
                v[j] = 0.95 * v[j] + 0.05 * g * g;
                let mhat = m[j] / (1.0 - 0.9_f32.powi((t + 1) as i32));
                let vhat = v[j] / (1.0 - 0.95_f32.powi((t + 1) as i32));
                w[j] = (1.0 - rate * 0.1) * w[j] - rate * mhat / (vhat.sqrt() + 1e-8);
            }
            for (actual, expected) in [
                (e.weights()?, &w),
                (opt.first.read()?, &m),
                (opt.second.read()?, &v),
            ] {
                assert!(
                    actual
                        .iter()
                        .zip(expected)
                        .all(|(a, b)| (a - b).abs() < 2e-6)
                );
            }
        }
        Ok(())
    }
}

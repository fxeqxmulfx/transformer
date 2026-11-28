//! Bias-corrected uncentered RMSProp with decoupled weight decay. Only the
//! mean square of each globally clipped gradient coordinate is retained.
use super::{Engine, F, Result, adamw::RatePlan, grid, page::Observations};
use crate::schedule::Schedule;
use serde_json::{Value, json};

pub(crate) struct RmsProp {
    mean_square: F,
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

impl RmsProp {
    pub fn new(engine: &Engine, schedule: Schedule, scale: f64, decay: f64) -> Result<Self> {
        Ok(Self {
            mean_square: engine.cuda.upload(&vec![0.0; engine.parameters.len()])?,
            schedule,
            scale,
            decay,
            completed: 0,
        })
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
        let bias = (1.0 - 0.95_f64.powf((self.completed + 1) as f64)).sqrt() as f32;
        // SAFETY: one thread per coordinate in equally sized weights, gradients and EMA buffers.
        unsafe {
            engine.kernel.rmsprop_decay(
                grid(p.len()),
                p.value.ptr(),
                self.mean_square.ptr(),
                p.grad.ptr(),
                norm.ptr(),
                p.len() as u32,
                rate,
                1.0 - rate * self.decay as f32,
                bias,
            )?;
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
}

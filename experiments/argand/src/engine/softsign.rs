//! Lion with a continuous softsign direction c/(|c|+epsilon).
//! Every independent real coordinate, including gates and scores, updates.
use super::{Engine, F, Result, adamw::RatePlan, grid, page::Observations};
use crate::schedule::Schedule;
use serde_json::{Value, json};

const EPSILON: f32 = 1e-6;

pub(crate) struct Softsign {
    moment: F,
    row_clip: Option<super::row_clip::RowClip>,
    schedule: Schedule,
    scale: f64,
    decay: f64,
    completed: usize,
}

pub(crate) fn settings(scale: f64, decay: f64, row_norm: Option<f64>, schedule: Schedule) -> Value {
    let mut settings = json!({"peak":crate::schedule::PEAK*scale,"min_rate":1e-8*scale,"budget":schedule.budget,
        "warmup":schedule.warmup,"cooldown_start":schedule.cooldown_start,"weight_decay":decay,
        "direction_beta":0.9,"moment_beta":0.99,"epsilon":1e-6,"clip_norm":1,"direction":"c/(abs(c)+epsilon)"});
    if let Some(radius) = row_norm {
        settings["max_projection_row_norm"] = radius.into();
    }
    settings
}

impl Softsign {
    pub fn new(
        engine: &Engine,
        schedule: Schedule,
        scale: f64,
        decay: f64,
        row_norm: Option<f64>,
    ) -> Result<Self> {
        Ok(Self {
            row_clip: row_norm
                .map(|r| super::row_clip::RowClip::new(engine, r as f32))
                .transpose()?,
            moment: engine.cuda.upload(&vec![0.0; engine.parameters.len()])?,
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
        // SAFETY: each thread owns one element in each equal-sized buffer.
        unsafe {
            engine.kernel.softsign_momentum(
                grid(p.len()),
                p.value.ptr(),
                self.moment.ptr(),
                p.grad.ptr(),
                norm.ptr(),
                p.len() as u32,
                rate,
                self.decay as f32,
                EPSILON,
            )?;
        }
        if let Some(clip) = &self.row_clip {
            clip.apply(engine)?;
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
    fn both_momenta_use_the_previous_buffer_and_clipped_gradient() -> Result<()> {
        let shape = crate::suite::Shape {
            width: 16,
            heads: 2,
            layers: 1,
            vocab: 4,
            max_batch: 2,
            max_length: 2,
        };
        let mut engine = Engine::new(shape, 1, 3)?;
        let schedule = super::super::adamw::wsd(20);
        let mut opt = Softsign::new(&engine, schedule, 0.3, 0.65, None)?;
        let mut expected = engine.weights()?;
        let mut moment = 0.0_f32;
        for (i, raw) in [3.0_f32, 0.0, -2.0, 1e-6].into_iter().enumerate() {
            let mut grads = vec![0.0; expected.len()];
            grads[0] = raw;
            engine.parameters.grad.write(&grads)?;
            opt.apply(&engine)?;
            let g = raw / raw.abs().max(1.0);
            let dir = 0.9 * moment + 0.1 * g;
            moment = 0.99 * moment + 0.01 * g;
            let rate = RatePlan::Scaled(0.3).rate(schedule, i);
            expected[0] = (1.0 - rate * 0.65) * expected[0] - rate * dir / (dir.abs() + EPSILON);
            assert!((opt.moment.read()?[0] - moment).abs() < 1e-8);
            assert!((engine.weights()?[0] - expected[0]).abs() < 2e-7);
        }
        Ok(())
    }
}

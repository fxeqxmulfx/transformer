//! Stopping rule of a sequence-accuracy case in the frozen 165-case GPT plan.
use serde_json::Value;

#[derive(Clone, Copy)]
pub struct SourceTarget {
    pub threshold: f64,
    pub after_target: usize,
}

impl SourceTarget {
    pub fn from_case(case: &Value) -> Result<Self, &'static str> {
        if case["metric"] != "sequence_accuracy" {
            return Err("source target requires a sequence-accuracy scorer");
        }
        Ok(Self {
            threshold: case["target"]
                .as_f64()
                .filter(|&target| target > 0. && target <= 1.)
                .ok_or("invalid source accuracy target")?,
            after_target: case["after_target"]
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or("missing source retention window")?,
        })
    }

    pub fn met(self, correct: u64, total: u64) -> bool {
        total > 0 && correct as f64 / total as f64 >= self.threshold
    }

    pub fn stop_after_crossing(self, step: usize, observe: usize, budget: usize) -> usize {
        step.saturating_add(self.after_target)
            .div_ceil(observe)
            .saturating_mul(observe)
            .min(budget)
    }
}

//! First-crossing and retention rules from the frozen 165-case GPT plan.
use serde_json::Value;

#[derive(Clone, Copy)]
enum Metric {
    Accuracy,
    NegativeMse,
}

#[derive(Clone, Copy)]
pub struct SourceTarget {
    pub threshold: f64,
    pub after_target: usize,
    metric: Metric,
}

impl SourceTarget {
    pub fn from_case(case: &Value) -> Result<Self, &'static str> {
        let threshold = case["target"].as_f64().ok_or("missing source target")?;
        let metric = match case["metric"].as_str() {
            Some("sequence_accuracy") if threshold > 0. && threshold <= 1. => Metric::Accuracy,
            Some("negative_mse") if threshold.is_finite() && threshold <= 0. => Metric::NegativeMse,
            _ => return Err("unsupported source metric or invalid threshold"),
        };
        Ok(Self {
            metric,
            threshold,
            after_target: case["after_target"]
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or("missing source retention window")?,
        })
    }

    pub fn met(self, correct: u64, total: u64, mse: Option<f64>) -> Result<bool, &'static str> {
        match (self.metric, mse) {
            (Metric::Accuracy, None) => {
                Ok(total > 0 && correct as f64 / total as f64 >= self.threshold)
            }
            (Metric::NegativeMse, Some(value)) if value.is_finite() && value >= 0. => {
                Ok(-value >= self.threshold)
            }
            _ => Err("observation differs from the declared source metric"),
        }
    }

    pub fn stop_after_crossing(self, step: usize, observe: usize, budget: usize) -> usize {
        step.saturating_add(self.after_target)
            .div_ceil(observe)
            .saturating_mul(observe)
            .min(budget)
    }
}

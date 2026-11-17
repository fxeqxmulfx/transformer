//! CM109 at1bb685d2: exact complete-class finite CE reuse, in guarded f64.
//! Shares normalize every original class. Forward training uses finite loss
//! differences, never the analytic residual used by the separate Mini branch.
use crate::Result;
pub struct Centre {
    pub loss: f64,
    pub shares: Vec<f64>,
    pub target: usize,
}
impl Centre {
    pub fn new(logits: &[f64], target: usize) -> Result<Self> {
        let loss = bilinear::model::cross_entropy(logits, target)?;
        let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut shares: Vec<_> = logits.iter().map(|x| (x - max).exp()).collect();
        let sum: f64 = shares.iter().sum();
        for p in &mut shares {
            *p /= sum;
        }
        Ok(Self {
            loss,
            shares,
            target,
        })
    }
    pub fn increment(&self, class: usize, delta: f64) -> Result<f64> {
        if !delta.is_finite() {
            return Err("nonfinite one-class query".into());
        }
        let share = *self
            .shares
            .get(class)
            .ok_or("query outside original vocabulary")?;
        let change = share * delta.exp_m1();
        if !change.is_finite() || change <= -1. {
            return Err("rounded cached CE factor is not positive".into());
        }
        let value = change.ln_1p() - if class == self.target { delta } else { 0. };
        if !value.is_finite() {
            return Err("nonfinite cached CE change".into());
        }
        Ok(value)
    }
    pub fn difference(&self, class: usize, plus: f64, minus: f64) -> Result<f64> {
        Ok(self.increment(class, plus)? - self.increment(class, minus)?)
    }
}

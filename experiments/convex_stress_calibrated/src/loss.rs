//! CCa1: ordinary softmax CE of log((1-epsilon)Q + epsilon/V) equals
//! negative log target probability and is convex in the joint coefficients.
use crate::{Residual, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Readout {
    Scaled(f64),
    Calibrated(f64),
}
impl Readout {
    pub fn validate(self) -> Result<()> {
        match self {
            Self::Scaled(s) if s.is_finite() && s > 0. => Ok(()),
            Self::Calibrated(e) if e.is_finite() && e > 0. && e < 1. => Ok(()),
            _ => Err("invalid fixed scale or smoothing probability".into()),
        }
    }
    pub fn logits(self, q: &[f64]) -> Vec<f64> {
        match self {
            Self::Scaled(s) => q.iter().map(|x| s * x).collect(),
            Self::Calibrated(e) => q
                .iter()
                .map(|x| ((1. - e) * x + e / q.len() as f64).ln())
                .collect(),
        }
    }
    pub fn ideal_ce(self, vocab: usize) -> f64 {
        match self {
            Self::Scaled(s) => ((vocab as f64 - 1.) * (-s).exp()).ln_1p(),
            Self::Calibrated(e) => -((1. - e) + e / vocab as f64).ln(),
        }
    }
    /// The scaled branch keeps the original class-constant centering exactly.
    /// It does not change feasible directional derivatives or the FW gap.
    pub(crate) fn residual(self, q: &[f64], label: usize) -> (Residual, f64) {
        match self {
            Self::Scaled(scale) => {
                let minimum = q.iter().copied().fold(f64::INFINITY, f64::min);
                let shifted: Vec<_> = q.iter().map(|x| scale * (x - minimum)).collect();
                let denom: f64 = shifted.iter().map(|x| x.exp()).sum();
                let mut r: Residual = shifted
                    .iter()
                    .enumerate()
                    .filter_map(|(c, &x)| {
                        let value = scale * x.exp_m1() / denom;
                        (value != 0.).then_some((c, value))
                    })
                    .collect();
                *r.entry(label).or_default() -= scale;
                (r, denom.ln() - shifted[label])
            }
            Self::Calibrated(e) => {
                let p = (1. - e) * q[label] + e / q.len() as f64;
                ([(label, -(1. - e) / p)].into(), -p.ln())
            }
        }
    }
}

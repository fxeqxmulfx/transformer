//! CM39 batch_fderiv_eq at 2ca6d2cf. Formula ports coverage_extension at
//! 08e4108b, with explicit original record identities and the zero derivative
//! in every supplemental coordinate. Repeated inputs retain their weights.
use crate::*;
use std::collections::BTreeSet;

pub struct Gradient {
    pub loss: f64,
    pub prior: Vec<f64>,
    pub supplement: Vec<f64>,
    pub scalar_work_forecast: u64,
}

impl Model {
    /// Positions in the immutable original record list, including a final tail.
    pub fn batch_gradient(&self, positions: &[usize]) -> Result<Gradient> {
        if positions.is_empty()
            || positions.len() > self.records.len()
            || positions.iter().copied().collect::<BTreeSet<_>>().len() != positions.len()
            || positions.iter().any(|&i| i >= self.records.len())
        {
            return Err("invalid original record batch positions".into());
        }
        let storage =
            self.prior.coefficients.len() as u64 + self.supplement.coefficients.len() as u64;
        let work = positions.iter().try_fold(storage, |n, &i| {
            let (b, _) = self.prior.location(self.records[i].cell);
            n.checked_add(16 * (self.prior.blocks[b].states() * self.prior.vocabulary) as u64)
                .ok_or("batch work overflow")
        })?;
        if work > bounds::WORK_CAP {
            return Err("extension derivative exceeds 32B work forecast".into());
        }
        let mut gradient = vec![0.; self.prior.coefficients.len()];
        let mut loss = 0.;
        for &i in positions {
            let record = &self.records[i];
            let (block, state) = self.prior.location(record.cell);
            let b = &self.prior.blocks[block];
            let logits = self.logits_cell(record.cell)?;
            loss += shared::training::ce(&logits, record.label);
            let top = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let sum: f64 = logits.iter().map(|x| (x - top).exp()).sum();
            for (c, &logit) in logits.iter().enumerate() {
                let delta = ((logit - top).exp() / sum - f64::from(c == record.label))
                    / positions.len() as f64;
                for f in 0..b.states() {
                    let phi = b.code[state * b.states() + f];
                    let h = 2 * (b.offset + f) * self.prior.vocabulary;
                    gradient[h + c] += phi.re * delta;
                    gradient[h + self.prior.vocabulary + c] -= phi.im * delta;
                }
            }
        }
        Ok(Gradient {
            loss: loss / positions.len() as f64,
            prior: gradient,
            supplement: vec![0.; self.supplement.coefficients.len()],
            scalar_work_forecast: work,
        })
    }
}

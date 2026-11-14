//! Actual raw-CE derivative in the free prior complex coordinates, CM39.
//! Formula ported from kernel/training.rs at2ca6d2cf, with original record-ID
//! multiplicity retained instead of rejecting repeated categorical cells.
use crate::*;
use std::collections::BTreeSet;

pub struct Gradient {
    pub loss: f64,
    pub prior: Vec<f64>,
    pub scalar_work_forecast: u64,
}

impl Model {
    pub fn batch_gradient(&self, ids: &[usize]) -> Result<Gradient> {
        if ids.is_empty()
            || ids.len() > self.records.len()
            || ids.iter().copied().collect::<BTreeSet<_>>().len() != ids.len()
            || ids.iter().any(|&i| i >= self.records.len())
        {
            return Err("invalid original record-ID batch".into());
        }
        let work = ids
            .iter()
            .try_fold(self.prior.coefficients.len() as u64, |n, &i| {
                let (b, _) = self.prior.location(self.records[i][0]);
                n.checked_add(16 * (self.prior.blocks[b].states() * self.prior.vocabulary) as u64)
                    .ok_or("batch work overflow")
            })?;
        if work > bounds::WORK_CAP {
            return Err("extension derivative exceeds32B work forecast".into());
        }
        let mut gradient = vec![0.; self.prior.coefficients.len()];
        let mut loss = 0.;
        for &i in ids {
            let [cell, label] = self.records[i];
            let (block, state) = self.prior.location(cell);
            let b = &self.prior.blocks[block];
            let logits = self.logits_cell(cell)?;
            loss += kernel::training::ce(&logits, label);
            let top = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let sum: f64 = logits.iter().map(|x| (x - top).exp()).sum();
            for (c, &logit) in logits.iter().enumerate() {
                let delta = ((logit - top).exp() / sum - f64::from(c == label)) / ids.len() as f64;
                for f in 0..b.states() {
                    let phi = b.code[state * b.states() + f];
                    let h = 2 * (b.offset + f) * self.prior.vocabulary;
                    gradient[h + c] += phi.re * delta;
                    gradient[h + self.prior.vocabulary + c] -= phi.im * delta;
                }
            }
        }
        Ok(Gradient {
            loss: loss / ids.len() as f64,
            prior: gradient,
            scalar_work_forecast: work,
        })
    }
}

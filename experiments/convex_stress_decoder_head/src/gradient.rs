//! Ordinary CE derivatives of every actual real and imaginary coefficient.
//! Original record positions form a finite batch; repeated inputs retain weight.
use crate::*;
use std::collections::BTreeSet;

#[derive(Debug)]
pub struct Gradient {
    pub loss: f64,
    pub coefficients: Vec<f64>,
    pub scalar_work_forecast: u64,
}

impl Model {
    pub fn batch_gradient(&self, positions: &[usize]) -> Result<Gradient> {
        if positions.is_empty()
            || positions.len() > self.records.len()
            || positions.iter().copied().collect::<BTreeSet<_>>().len() != positions.len()
            || positions.iter().any(|&i| i >= self.records.len())
        {
            return Err("invalid original record positions".into());
        }
        let scalar_work_forecast = self.coefficients.len() as u64
            + 16 * (positions.len() * self.source.q * self.source.vocab) as u64;
        if scalar_work_forecast > WORK_CAP {
            return Err("complete batch gradient exceeds work limit".into());
        }
        let mut coefficients = vec![0.; self.coefficients.len()];
        let mut loss = 0.;
        for &i in positions {
            let r = &self.records[i];
            let phi = self.features(r.cell)?;
            let z = self.logits_cell(r.cell)?;
            loss += tagged::loss(&z, r.label);
            let top = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let total: f64 = z.iter().map(|x| (x - top).exp()).sum();
            for (c, &logit) in z.iter().enumerate() {
                let delta = ((logit - top).exp() / total - f64::from(c == r.label))
                    / positions.len() as f64;
                for (f, p) in phi.iter().enumerate() {
                    let h = 2 * f * self.source.vocab;
                    coefficients[h + c] += p.re * delta;
                    coefficients[h + self.source.vocab + c] -= p.im * delta;
                }
            }
        }
        Ok(Gradient {
            loss: loss / positions.len() as f64,
            coefficients,
            scalar_work_forecast,
        })
    }
}

//! CM42 exact original-batch derivatives. All real and imaginary coordinates
//! are returned, and original record IDs preserve repeated examples and tails.
use crate::*;

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
        let work =
            positions
                .iter()
                .try_fold(self.source.coefficients.len() as u64, |work, &i| {
                    let (b, _) = self.source.location(self.records[i].cell);
                    work.checked_add(
                        16 * (self.source.blocks[b].states() * self.source.vocabulary) as u64,
                    )
                    .ok_or("context batch work overflow")
                })?;
        if work > kernel::limits::WORK_CAP {
            return Err("context batch exceeds work limit".into());
        }
        let mut coefficients = vec![0.; self.source.coefficients.len()];
        let mut loss = 0.;
        for &i in positions {
            let r = &self.records[i];
            let (block, phi) = self.features(r.cell)?;
            let b = &self.source.blocks[block];
            let logits = self.logits_cell(r.cell)?;
            loss += kernel::training::ce(&logits, r.label);
            let top = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let total: f64 = logits.iter().map(|z| (z - top).exp()).sum();
            let v = self.source.vocabulary;
            for (c, &z) in logits.iter().enumerate() {
                let delta =
                    ((z - top).exp() / total - f64::from(c == r.label)) / positions.len() as f64;
                for (f, p) in phi.iter().enumerate() {
                    let h = 2 * (b.offset + f) * v;
                    coefficients[h + c] += p.re * delta;
                    coefficients[h + v + c] -= p.im * delta;
                }
            }
        }
        Ok(Gradient {
            loss: loss / positions.len() as f64,
            coefficients,
            scalar_work_forecast: work,
        })
    }
}

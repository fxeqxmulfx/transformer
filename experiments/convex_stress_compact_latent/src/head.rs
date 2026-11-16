//! CM70 complete complex head, after Lean cc6931a4. This is the arithmetic
//! of reduced_latent at 6797e8ab with lossless shared column storage. Unknown
//! blocks still visit every class in every contraction and spectral audit.
use crate::domain::{self, Domain};
use crate::{Parameters, Result, Shape, Work};
pub use legacy::head::{
    Audit, Certificate, Prediction, PredictionBudget, reference_source, source,
};
mod storage;
use storage::{ALL, Block, Builder, Exception, Support};
mod exact;
#[path = "../../convex_stress_reduced_latent/src/head/interval.rs"]
mod interval;
mod predict;
#[cfg(test)]
mod tests;

#[derive(Debug)]
pub struct Head {
    shape: Shape,
    blocks: Vec<Block>,
    exceptions: Vec<Exception>,
    active: Vec<u16>,
}
impl Head {
    /// The limit covers this head and construction buffers; the caller must
    /// additionally charge input parameters, dictionaries and query scratch.
    pub fn build(domain: &Domain, q: &Parameters, limit: u64, work: &mut Work) -> Result<Self> {
        domain::validate(domain, work)?;
        q.validate(domain)?;
        work.charge(160 * domain.head_columns_bound as u128 + 16 * domain.shape.rows() as u128)?;
        let sh = domain.shape;
        let mut out = Builder::new(sh, limit)?;
        for row in (0..sh.rows()).step_by(4) {
            let (p, base) = (row / sh.states, row % sh.states);
            let support = Support::new(domain, row);
            out.scratch[..support.len()].fill([0.; 8]);
            for f in 0..4 {
                for s in base..base + 4 {
                    let (re, im) = source(s, base + f);
                    for (i, c) in support.iter().enumerate() {
                        let g = q.gamma(domain, p, s, c);
                        out.scratch[i][f] += re * g;
                        out.scratch[i][4 + f] -= im * g;
                    }
                }
            }
            out.push(&support)?;
        }
        out.finish()
    }
    pub fn shape(&self) -> Shape {
        self.shape
    }
    pub fn stored_coefficients(&self) -> usize {
        (self.blocks.len() + self.exceptions.len()) * 8
    }
    pub fn coefficient(&self, p: usize, f: usize, c: usize) -> Result<(f64, f64)> {
        if p >= self.shape.programs || f >= self.shape.states || c >= self.shape.classes {
            return Err("invalid complete complex coefficient index".into());
        }
        let b = &self.blocks[(p * self.shape.states + f) / 4];
        if b.active_start != ALL
            && self.active[b.active_start..b.active_start + b.active_len]
                .binary_search(&(c as u16))
                .is_err()
        {
            return Ok((0., 0.));
        }
        let value = self.column(b, c);
        Ok((value[f % 4], value[4 + f % 4]))
    }
    fn query(&self, states: &[usize]) -> Result<usize> {
        if states.len() != self.shape.programs || states.iter().any(|&s| s >= self.shape.states) {
            return Err("invalid input-only reduced latent states".into());
        }
        Ok(states
            .iter()
            .enumerate()
            .map(|(p, &s)| self.blocks[(p * self.shape.states + s) / 4].active_len)
            .sum())
    }
    pub fn logits(&self, states: &[usize], work: &mut Work) -> Result<Vec<f64>> {
        let columns = self.query(states)?;
        work.charge(40 * columns as u128 + 8 * self.shape.classes as u128)?;
        let mut logits = vec![0.; self.shape.classes];
        for (p, &s) in states.iter().enumerate() {
            let b = &self.blocks[(p * self.shape.states + s) / 4];
            for f in 0..4 {
                let (re, im) = source(s, s / 4 * 4 + f);
                for (c, value) in self.columns(b) {
                    logits[c] += re * value[f] - im * value[4 + f];
                }
            }
        }
        if logits.iter().any(|x| !x.is_finite()) {
            return Err("nonfinite complete sparse logit".into());
        }
        Ok(logits)
    }
    pub fn audit(&self, domain: &Domain, q: &Parameters, work: &mut Work) -> Result<Audit> {
        domain::validate(domain, work)?;
        q.validate(domain)?;
        self.validate_support(domain)?;
        work.charge(400 * domain.head_columns_bound as u128 + 32 * self.shape.rows() as u128)?;
        let mut column_l1 = vec![0.; self.shape.classes];
        let mut residual = 0.0f64;
        for (bi, b) in self.blocks.iter().enumerate() {
            let (p, base) = (bi * 4 / self.shape.states, bi * 4 % self.shape.states);
            for (c, value) in self.columns(b) {
                for f in 0..4 {
                    if !value[f].is_finite() || !value[4 + f].is_finite() {
                        return Err("nonfinite stored complex coefficient".into());
                    }
                    column_l1[c] += value[f].abs() + value[4 + f].abs();
                }
                for s in base..base + 4 {
                    let (mut re, mut im) = (0., 0.);
                    for f in (0..4).rev() {
                        let (u, v) = reference_source(s, base + f);
                        re += (u as f64 * value[f] - v as f64 * value[4 + f]) * 0.5;
                        im += (u as f64 * value[4 + f] + v as f64 * value[f]) * 0.5;
                    }
                    residual = residual
                        .max((re - q.gamma(domain, p, s, c)).abs())
                        .max(im.abs());
                }
            }
        }
        let bound = 2. * domain.maximum_mass * self.shape.states as f64;
        if !residual.is_finite()
            || residual > 1e-7
            || column_l1
                .iter()
                .any(|&x| !x.is_finite() || x > bound + 1e-7)
        {
            return Err("complete head inverse residual or spectral budget failed".into());
        }
        Ok(Audit {
            column_l1,
            maximum_inverse_residual: residual,
            logical_coefficients: self.shape.cells() * 2,
            stored_coefficients: self.stored_coefficients(),
        })
    }
}

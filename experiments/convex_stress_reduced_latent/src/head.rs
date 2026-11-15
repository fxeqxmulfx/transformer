//! Complete CM64 block-Fourier coefficients with only proved zero columns
//! implicit. Port of bounded_latent at a3959a46. Each stored coefficient uses
//! the same four binary64 operations as its dense counterpart. Unknown rows
//! always retain every class. CM65 changes storage, not these rounded products.
use crate::{domain::Domain, *};
pub use dense::head::{reference_source, source};
mod exact;
mod interval;
mod predict;
pub use predict::{Certificate, Prediction, PredictionBudget};

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Block {
    pub classes: Vec<usize>,
    pub real: Vec<[f64; 4]>,
    pub imaginary: Vec<[f64; 4]>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct Head {
    pub shape: Shape,
    pub(crate) blocks: Vec<Block>,
}

fn classes(domain: &Domain, row: usize) -> Vec<usize> {
    let labels = &domain.labels[row..row + 4];
    if labels.iter().any(Option::is_none) {
        return (0..domain.shape.classes).collect();
    }
    let mut out: Vec<_> = labels.iter().flatten().copied().collect();
    out.sort_unstable();
    out.dedup();
    out
}
impl Head {
    pub fn build(domain: &Domain, q: &Parameters, work: &mut Work) -> Result<Self> {
        domain.validate(work)?;
        q.validate(domain)?;
        work.charge(160 * domain.head_columns_bound as u128 + 16 * domain.shape.rows() as u128)?;
        let sh = domain.shape;
        let mut blocks = Vec::with_capacity(sh.rows() / 4);
        for row in (0..sh.rows()).step_by(4) {
            let (p, base) = (row / sh.states, row % sh.states);
            let classes = classes(domain, row);
            let mut block = Block {
                real: vec![[0.; 4]; classes.len()],
                imaginary: vec![[0.; 4]; classes.len()],
                classes,
            };
            for f in 0..4 {
                for s in base..base + 4 {
                    let (re, im) = source(s, base + f);
                    for (i, &c) in block.classes.iter().enumerate() {
                        let g = q.gamma(domain, p, s, c);
                        block.real[i][f] += re * g;
                        block.imaginary[i][f] -= im * g;
                    }
                }
            }
            blocks.push(block);
        }
        Ok(Self { shape: sh, blocks })
    }
    pub fn stored_coefficients(&self) -> usize {
        self.blocks.iter().map(|b| b.classes.len() * 8).sum()
    }
    pub fn coefficient(&self, p: usize, f: usize, c: usize) -> Result<(f64, f64)> {
        if p >= self.shape.programs || f >= self.shape.states || c >= self.shape.classes {
            return Err("invalid complete complex coefficient index".into());
        }
        let b = &self.blocks[(p * self.shape.states + f) / 4];
        Ok(match b.classes.binary_search(&c) {
            Ok(i) => (b.real[i][f % 4], b.imaginary[i][f % 4]),
            Err(_) => (0., 0.),
        })
    }
    fn query(&self, states: &[usize]) -> Result<usize> {
        if states.len() != self.shape.programs || states.iter().any(|&s| s >= self.shape.states) {
            return Err("invalid input-only reduced latent states".into());
        }
        Ok(states
            .iter()
            .enumerate()
            .map(|(p, &s)| self.blocks[(p * self.shape.states + s) / 4].classes.len())
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
                for (i, &c) in b.classes.iter().enumerate() {
                    logits[c] += re * b.real[i][f] - im * b.imaginary[i][f];
                }
            }
        }
        if logits.iter().any(|x| !x.is_finite()) {
            return Err("nonfinite complete sparse logit".into());
        }
        Ok(logits)
    }
    /// Includes every logical coefficient. Implicit columns are exact zeros
    /// because all four gamma entries are anchored at other classes.
    pub fn audit(&self, domain: &Domain, q: &Parameters, work: &mut Work) -> Result<Audit> {
        domain.validate(work)?;
        q.validate(domain)?;
        if self.shape != domain.shape || self.blocks.len() != self.shape.rows() / 4 {
            return Err("head dimensions differ from full logical tensor".into());
        }
        work.charge(400 * domain.head_columns_bound as u128 + 32 * self.shape.rows() as u128)?;
        let mut column_l1 = vec![0.; self.shape.classes];
        let mut residual = 0.0f64;
        for (bi, b) in self.blocks.iter().enumerate() {
            if b.classes != classes(domain, bi * 4)
                || b.real.len() != b.classes.len()
                || b.imaginary.len() != b.classes.len()
            {
                return Err("noncanonical sparse head support".into());
            }
            let (p, base) = (bi * 4 / self.shape.states, bi * 4 % self.shape.states);
            for (i, &c) in b.classes.iter().enumerate() {
                for f in 0..4 {
                    if !b.real[i][f].is_finite() || !b.imaginary[i][f].is_finite() {
                        return Err("nonfinite stored complex coefficient".into());
                    }
                    column_l1[c] += b.real[i][f].abs() + b.imaginary[i][f].abs();
                }
                for s in base..base + 4 {
                    let (mut re, mut im) = (0., 0.);
                    for f in (0..4).rev() {
                        let (u, v) = reference_source(s, base + f);
                        re += (u as f64 * b.real[i][f] - v as f64 * b.imaginary[i][f]) * 0.5;
                        im += (u as f64 * b.imaginary[i][f] + v as f64 * b.real[i][f]) * 0.5;
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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Audit {
    pub column_l1: Vec<f64>,
    pub maximum_inverse_residual: f64,
    pub logical_coefficients: usize,
    pub stored_coefficients: usize,
}

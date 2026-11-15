//! Complete CM64 head H = U* gamma for the block Fourier source proved at
//! aa1d0371. Exact source zeros allow four products per queried program;
//! every learned coefficient is stored and included in the column audit.
use crate::*;
mod exact;
mod interval;
mod predict;
pub use predict::{Certificate, Prediction, PredictionBudget};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Head {
    pub shape: Shape,
    pub real: Vec<f64>,
    pub imaginary: Vec<f64>,
}

/// Positive Fourier phase, matching LatentPrograms.fourier4 in Lean.
pub fn source(row: usize, frequency: usize) -> (f64, f64) {
    if row / 4 != frequency / 4 {
        return (0., 0.);
    }
    match (row % 4) * (frequency % 4) % 4 {
        0 => (0.5, 0.),
        1 => (0., 0.5),
        2 => (-0.5, 0.),
        _ => (0., -0.5),
    }
}

/// Independent integer phase table used by the second contraction.
pub fn reference_source(row: usize, frequency: usize) -> (i8, i8) {
    const REAL: [[i8; 4]; 4] = [[1, 1, 1, 1], [1, 0, -1, 0], [1, -1, 1, -1], [1, 0, -1, 0]];
    const IMAG: [[i8; 4]; 4] = [[0, 0, 0, 0], [0, 1, 0, -1], [0, 0, 0, 0], [0, -1, 0, 1]];
    if row / 4 != frequency / 4 {
        (0, 0)
    } else {
        (REAL[row % 4][frequency % 4], IMAG[row % 4][frequency % 4])
    }
}

impl Head {
    pub fn at(&self, p: usize, f: usize, c: usize) -> usize {
        (p * self.shape.states + f) * self.shape.classes + c
    }
    pub fn validate(&self) -> Result<()> {
        self.shape.admit(0)?;
        if self.real.len() != self.shape.cells()
            || self.imaginary.len() != self.shape.cells()
            || self
                .real
                .iter()
                .chain(&self.imaginary)
                .any(|x| !x.is_finite() || x.abs() > 1e13)
        {
            return Err("invalid complete block Fourier head".into());
        }
        Ok(())
    }
    pub fn build(shape: Shape, q: &Parameters, work: &mut Work) -> Result<Self> {
        q.validate(shape)?;
        work.charge(40 * shape.cells() as u128)?;
        let mut head = Self {
            shape,
            real: vec![0.; shape.cells()],
            imaginary: vec![0.; shape.cells()],
        };
        for p in 0..shape.programs {
            for f in 0..shape.states {
                for s in (f / 4 * 4)..(f / 4 * 4 + 4) {
                    let (re, im) = source(s, f);
                    for c in 0..shape.classes {
                        let i = head.at(p, f, c);
                        let g = q.values[shape.at(p, s, c)];
                        head.real[i] += re * g;
                        head.imaginary[i] -= im * g;
                    }
                }
            }
        }
        head.validate()?;
        Ok(head)
    }
    fn query(&self, states: &[usize]) -> Result<()> {
        if states.len() != self.shape.programs || states.iter().any(|&s| s >= self.shape.states) {
            return Err("invalid input-only latent states".into());
        }
        Ok(())
    }
    /// Ordinary floating products for training CE; no ideal decoder replacement.
    pub fn logits(&self, states: &[usize], work: &mut Work) -> Result<Vec<f64>> {
        self.query(states)?;
        work.charge(40 * self.shape.programs as u128 * self.shape.classes as u128)?;
        let mut logits = vec![0.; self.shape.classes];
        for (p, &s) in states.iter().enumerate() {
            for f in (s / 4 * 4)..(s / 4 * 4 + 4) {
                let (re, im) = source(s, f);
                for (c, z) in logits.iter_mut().enumerate() {
                    let i = self.at(p, f, c);
                    *z += re * self.real[i] - im * self.imaginary[i];
                }
            }
        }
        if logits.iter().any(|z| !z.is_finite()) {
            return Err("nonfinite complex-head logit".into());
        }
        Ok(logits)
    }
    /// Read every saved coefficient, including ones zero at the current query.
    pub fn audit(&self, q: &Parameters, work: &mut Work) -> Result<HeadAudit> {
        self.validate()?;
        q.validate(self.shape)?;
        work.charge(80 * self.shape.cells() as u128)?;
        let sh = self.shape;
        let mut residual = 0.0f64;
        let mut column_l1 = vec![0.; sh.classes];
        for p in 0..sh.programs {
            for s in 0..sh.states {
                for (c, norm) in column_l1.iter_mut().enumerate() {
                    let i = self.at(p, s, c);
                    *norm += self.real[i].abs() + self.imaginary[i].abs();
                    let (mut re, mut im) = (0., 0.);
                    for f in (s / 4 * 4..s / 4 * 4 + 4).rev() {
                        let (u, v) = reference_source(s, f);
                        let j = self.at(p, f, c);
                        re += (u as f64 * self.real[j] - v as f64 * self.imaginary[j]) * 0.5;
                        im += (u as f64 * self.imaginary[j] + v as f64 * self.real[j]) * 0.5;
                    }
                    residual = residual
                        .max((re - q.values[sh.at(p, s, c)]).abs())
                        .max(im.abs());
                }
            }
        }
        if !residual.is_finite() || residual > 1e-7 {
            return Err("complete coefficient inverse residual exceeds 1e-7".into());
        }
        Ok(HeadAudit {
            column_l1,
            maximum_inverse_residual: residual,
            coefficients: self.real.len() * 2,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HeadAudit {
    pub column_l1: Vec<f64>,
    pub maximum_inverse_residual: f64,
    pub coefficients: usize,
}

//! CM86 balanced learned query/context encoders and ordinary raw logits.
//! All missing storage columns mean exact zero, as proved in CM88 Support.
//! No class, complex component or small nonzero singular direction is dropped.
use crate::{
    C, Result, algebra,
    domain::{Addresses, Shape},
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Joint {
    pub shape: Shape,
    /// Context column -> every original class and every query address.
    pub columns: BTreeMap<usize, Vec<C>>,
}

impl Joint {
    pub fn zero(shape: Shape) -> Self {
        Self {
            shape,
            columns: BTreeMap::new(),
        }
    }

    pub fn entry(&self, class: usize, q: usize, d: usize) -> C {
        self.columns
            .get(&d)
            .map_or(C::default(), |col| col[class * self.shape.queries + q])
    }

    pub fn validate(&self, max_bytes: u64) -> Result<()> {
        if Shape::new(self.shape.objects, self.shape.context)? != self.shape
            || self.shape.reservation(self.columns.len())? > max_bytes
        {
            return Err("complete model shape or memory reservation refused".into());
        }
        let width = self
            .shape
            .objects
            .checked_mul(self.shape.queries)
            .ok_or("width overflow")?;
        for (&d, col) in &self.columns {
            if d >= self.shape.columns
                || col.len() != width
                || col.iter().any(|x| !x.re.is_finite() || !x.im.is_finite())
            {
                return Err("invalid complete complex column".into());
            }
        }
        Ok(())
    }

    /// Diagnostic only: CM86's proved joint-matrix identity.
    pub fn joint_complex(&self, input: &Addresses) -> Vec<C> {
        (0..self.shape.objects)
            .map(|c| {
                let mut out = C::default();
                for &(d, count) in &input.columns {
                    for &q in &input.queries {
                        out += self.entry(c, q, d) * count as f64;
                    }
                }
                C::new(1., 1.) * out
            })
            .collect()
    }
}

struct Factors {
    amplitude: f64,
    left: Vec<C>,
    right: Vec<C>,
}

pub struct Model {
    pub joint: Joint,
    stored: Vec<usize>,
    factors: Vec<Factors>,
    pub reconstruction_error: f64,
    pub balance_error: f64,
    pub max_bytes: u64,
}

impl Model {
    pub fn recover(joint: Joint, max_bytes: u64) -> Result<Self> {
        joint.validate(max_bytes)?;
        let shape = joint.shape;
        let q = shape.queries;
        let stored: Vec<_> = joint.columns.keys().copied().collect();
        let n = stored.len();
        let mut factors = Vec::with_capacity(shape.objects);
        let (mut reconstruction_error, mut balance_error) = (0f64, 0f64);
        for class in 0..shape.objects {
            let mut left = vec![C::default(); q * q];
            let mut right = vec![C::default(); n * q];
            let mut amplitude = 0.;
            if n > 0 {
                // BTreeMap values have the same order as `stored`. Read each
                // validated full class slice once instead of looking up its
                // column separately for every query coordinate.
                let matrix: Vec<_> = joint
                    .columns
                    .values()
                    .flat_map(|column| column[class * q..(class + 1) * q].iter().copied())
                    .collect();
                let decomposition = algebra::svd(q, n, &matrix)?;
                amplitude = decomposition.values.iter().sum();
                if !amplitude.is_finite() {
                    return Err("nonfinite learned readout".into());
                }
                if amplitude > 0. {
                    let norm = amplitude.sqrt();
                    let rank = decomposition.values.len();
                    for h in 0..rank {
                        let scale = decomposition.values[h].sqrt() / norm;
                        for i in 0..q {
                            left[i * q + h] = decomposition.u[h * q + i] * scale;
                        }
                        for d in 0..n {
                            right[d * q + h] = decomposition.vt[d * rank + h].conj() * scale;
                        }
                    }
                    let ln: f64 = left.iter().map(|x| x.norm_sqr()).sum();
                    let rn: f64 = right.iter().map(|x| x.norm_sqr()).sum();
                    balance_error = balance_error.max((ln - 1.).abs()).max((rn - 1.).abs());
                }
                for d in 0..n {
                    for i in 0..q {
                        let recovered: C = (0..q)
                            .map(|h| left[i * q + h] * right[d * q + h].conj())
                            .sum::<C>()
                            * amplitude;
                        let error = (recovered - matrix[d * q + i]).norm();
                        if !error.is_finite() || error > 2e-9 * (1. + matrix[d * q + i].norm()) {
                            return Err("numerical complete factor reconstruction refused".into());
                        }
                        reconstruction_error = reconstruction_error.max(error);
                    }
                }
            }
            factors.push(Factors {
                amplitude,
                left,
                right,
            });
        }
        if balance_error > 2e-9 {
            return Err("numerical balancedness refused".into());
        }
        Ok(Self {
            joint,
            stored,
            factors,
            reconstruction_error,
            balance_error,
            max_bytes,
        })
    }

    /// Actual raw forward: read both learned encoders, complex interaction,
    /// learned magnitude, then real projection to every original output class.
    pub fn complex(&self, input: &Addresses) -> Result<Vec<C>> {
        let q = self.joint.shape.queries;
        let context: Vec<_> = input
            .columns
            .iter()
            .filter_map(|&(d, n)| self.stored.binary_search(&d).ok().map(|i| (i, n)))
            .collect();
        let mut output = Vec::with_capacity(self.joint.shape.objects);
        for factors in &self.factors {
            let mut value = C::default();
            for h in 0..q {
                let query: C = input
                    .queries
                    .iter()
                    .map(|&i| factors.left[i * q + h].conj())
                    .sum();
                let context: C = context
                    .iter()
                    .map(|&(d, n)| factors.right[d * q + h].conj() * n as f64)
                    .sum();
                value += query.conj() * context;
            }
            value *= factors.amplitude * C::new(1., 1.);
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err("nonfinite actual learned forward".into());
            }
            output.push(value);
        }
        Ok(output)
    }

    pub fn logits(&self, prefix: &[i64]) -> Result<Vec<f64>> {
        let input = self.joint.shape.addresses(prefix)?;
        Ok(self.complex(&input)?.into_iter().map(|x| x.re).collect())
    }

    /// Diagnostic encoder row energies and learned output magnitude.
    pub fn geometry(&self, class: usize) -> Result<(f64, Vec<f64>, Vec<f64>)> {
        let factors = self
            .factors
            .get(class)
            .ok_or("class outside original vocabulary")?;
        let q = self.joint.shape.queries;
        Ok((
            factors.amplitude,
            factors
                .left
                .chunks_exact(q)
                .map(|r| r.iter().map(|z| z.norm_sqr()).sum())
                .collect(),
            factors
                .right
                .chunks_exact(q)
                .map(|r| r.iter().map(|z| z.norm_sqr()).sum())
                .collect(),
        ))
    }
}

pub fn cross_entropy(logits: &[f64], target: usize) -> Result<f64> {
    if target >= logits.len() || logits.iter().any(|x| !x.is_finite()) {
        return Err("invalid ordinary cross entropy input".into());
    }
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let loss = max - logits[target] + logits.iter().map(|x| (x - max).exp()).sum::<f64>().ln();
    if !loss.is_finite() || loss < -1e-12 {
        return Err("nonfinite ordinary CE".into());
    }
    Ok(loss)
}

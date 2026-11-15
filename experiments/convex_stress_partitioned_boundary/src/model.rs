//! The actual expanded affine head: a constant one-state block and all complete
//! real/imaginary Fourier coordinates in every latent block and raw class.
use crate::*;
use features::{Bank, Encoder, Row};
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Model {
    pub input: Input,
    pub protocol_sha256: String,
    pub bank: Bank,
    pub geometry: geometry::Geometry,
    pub partition: partition::Partition,
    pub common: f64,
    /// Binary order is minus,plus. First occurrence resolves exact centroid ties.
    pub decoder: Vec<Vec<usize>>,
    pub theta: Vec<f64>,
}
impl Model {
    pub fn weights(&self) -> (Vec<f64>, f64) {
        let mut offset = vec![0usize; self.input.vocabulary];
        let mut binary = offset.clone();
        for labels in &self.decoder {
            for &c in labels {
                offset[c] += 1;
                if labels.len() == 2 {
                    binary[c] += 1;
                }
            }
        }
        let m = self.bank.states();
        let mut w = vec![*binary.iter().max().unwrap() as f64; 2 * m];
        for j in 0..self.geometry.bank.dimension() * self.partition.leaves {
            let mut costs = vec![0.; self.input.vocabulary];
            for (b, labels) in self.decoder.iter().enumerate() {
                if labels.len() == 2
                    && self.partition.root.leaf(&self.geometry.descriptors[b])
                        == j / self.geometry.bank.dimension()
                {
                    for &c in labels {
                        costs[c] +=
                            self.geometry.descriptors[b][j % self.geometry.bank.dimension()].abs()
                                * (m as f64).sqrt().ceil();
                    }
                }
            }
            w.push(costs.into_iter().fold(0., f64::max));
        }
        (w, self.common * *offset.iter().max().unwrap() as f64)
    }
    pub fn validate(&self) -> Result<f64> {
        self.input.validate()?;
        self.partition.validate(self.geometry.bank.dimension())?;
        if self.protocol_sha256 != protocol_hash()?
            || self.decoder.len() != self.input.blocks
            || self.bank != Bank::Quadratic
            || self.theta.len()
                != 2 * self.bank.states() + self.geometry.bank.dimension() * self.partition.leaves
            || !COMMONS.contains(&self.common)
            || self.theta.iter().any(|x| !x.is_finite())
            || self.decoder.iter().any(|d| {
                d.len() > 2
                    || d.iter().any(|&c| c >= self.input.vocabulary)
                    || (d.len() == 2 && d[0] == d[1])
            })
        {
            return Err("invalid shared geometry model".into());
        }
        if self.geometry.descriptors.len() != self.input.blocks
            || self.geometry.descriptors.iter().any(|row| {
                row.len() != self.geometry.bank.dimension()
                    || row.iter().any(|x| !x.is_finite() || x.abs() > 1. + 1e-12)
            })
        {
            return Err("invalid input descriptor tensor".into());
        }
        let (w, t) = self.weights();
        if projection::mass(&self.theta, &w) > RADIUS - t + 1e-6 {
            return Err("weighted parameter budget exceeded".into());
        }
        let expanded = self.expanded_mass();
        if expanded > RADIUS * (1. + 1e-10) {
            return Err("expanded complete coefficient budget exceeded".into());
        }
        Ok(expanded)
    }
    pub fn radial(&mut self) -> Result<()> {
        let (w, t) = self.weights();
        let norm = projection::mass(&self.theta, &w);
        if t > RADIUS {
            return Err("fixed offset alone exceeds budget".into());
        }
        if norm > RADIUS - t {
            for x in &mut self.theta {
                *x *= (RADIUS - t) / norm;
            }
        }
        self.validate()?;
        Ok(())
    }
    pub fn project(&mut self) -> Result<f64> {
        let (w, t) = self.weights();
        projection::project(&mut self.theta, &w, RADIUS - t)
    }
    pub fn expanded_mass(&self) -> f64 {
        let m = self.bank.states();
        let mut masses = vec![0.; self.input.vocabulary];
        for (b, d) in self.decoder.iter().enumerate() {
            for &c in d {
                masses[c] += self.common;
                if d.len() == 2 {
                    for f in 0..m {
                        masses[c] += (self.theta[2 * f] + self.boundary(b) / (m as f64).sqrt())
                            .abs()
                            + self.theta[2 * f + 1].abs();
                    }
                }
            }
        }
        masses.into_iter().fold(0., f64::max)
    }
    pub fn logits_row(&self, row: &Row) -> Vec<f64> {
        let mut z = vec![0.; self.input.vocabulary];
        let d = &self.decoder[row.block];
        for &c in d {
            z[c] = self.common;
        }
        if d.len() == 2 {
            let q = self.contrast(row);
            z[d[0]] -= q;
            z[d[1]] += q;
        }
        z
    }
    pub fn contrast(&self, row: &Row) -> f64 {
        row.phi
            .iter()
            .enumerate()
            .map(|(f, p)| p.re * self.theta[2 * f] - p.im * self.theta[2 * f + 1])
            .sum::<f64>()
            + self.boundary(row.block) / row.scale
    }
    pub fn boundary(&self, block: usize) -> f64 {
        let d = self.geometry.bank.dimension();
        let leaf = self.partition.root.leaf(&self.geometry.descriptors[block]);
        self.theta[2 * self.bank.states() + leaf * d..2 * self.bank.states() + (leaf + 1) * d]
            .iter()
            .zip(&self.geometry.descriptors[block])
            .map(|(a, d)| a * d)
            .sum()
    }
    /// Exact full-vocabulary CE algebra, summing the identical zero-logit absent
    /// classes by multiplicity. Tests and the final audit compare the full loop.
    pub fn compact_score(&self, row: &Row) -> (f64, usize) {
        let labels = &self.decoder[row.block];
        let v = self.input.vocabulary as f64;
        match labels.as_slice() {
            [] => (v.ln(), 0),
            [c] => {
                let tail = ((v - 1.) * (-self.common).exp()).ln_1p();
                (
                    if row.label == *c {
                        tail
                    } else {
                        self.common + tail
                    },
                    *c,
                )
            }
            [minus, plus] => {
                let q = self.contrast(row);
                let a = q.abs();
                let tail = ((-2. * a).exp() + (v - 2.) * (-self.common - a).exp()).ln_1p();
                let loss = if row.label == *minus {
                    a + q + tail
                } else if row.label == *plus {
                    a - q + tail
                } else {
                    self.common + a + tail
                };
                let prediction = if q > 0. {
                    *plus
                } else if q < 0. {
                    *minus
                } else {
                    (*plus).min(*minus)
                };
                (loss, prediction)
            }
            _ => unreachable!("validated categorical decoder"),
        }
    }
    pub fn logits_cell(&self, cell: usize) -> Result<Vec<f64>> {
        if cell >= self.input.route.len() {
            return Err("query outside fixed route".into());
        }
        Ok(self.logits_row(&Encoder::new(self.bank).row(&self.input, cell, 0)))
    }
    /// Expand every coefficient and evaluate all actual complex products without
    /// using the compact q+beta/scale formula. Includes absent raw classes.
    pub fn independent_logits(&self, cell: usize) -> Result<Vec<f64>> {
        if cell >= self.input.route.len() {
            return Err("query outside fixed route".into());
        }
        let row = Encoder::new(self.bank).row(&self.input, cell, 0);
        let m = self.bank.states();
        let d = &self.decoder[row.block];
        let mut z = vec![0.; self.input.vocabulary];
        for (c, z) in z.iter_mut().enumerate() {
            if d.contains(&c) {
                *z = self.common;
            }
            for (f, p) in row.phi.iter().enumerate() {
                let sign = if d.len() == 2 && d[0] == c {
                    -1.
                } else if d.len() == 2 && d[1] == c {
                    1.
                } else {
                    0.
                };
                let mut re = sign * self.theta[2 * f];
                let start = self
                    .partition
                    .root
                    .leaf(&self.geometry.descriptors[row.block])
                    * self.geometry.bank.dimension();
                for (j, d) in self.geometry.descriptors[row.block].iter().enumerate() {
                    re += sign * d * self.theta[2 * m + start + j] / (m as f64).sqrt();
                }
                let im = sign * self.theta[2 * f + 1];
                *z += p.re * re - p.im * im;
            }
        }
        Ok(z)
    }
}

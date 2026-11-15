//! Complete CM57 expansion and actual complex products, Head.lean at258dd055.
//! The five decoder coefficients and mass are already joint affine parameters.
use crate::*;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

fn code(s: usize, f: usize) -> Complex64 {
    match (s * f) % 4 {
        0 => Complex64::new(0.5, 0.),
        1 => Complex64::new(0., 0.5),
        2 => Complex64::new(-0.5, 0.),
        _ => Complex64::new(0., -0.5),
    }
}
fn source(k: usize, s: usize) -> f64 {
    let c = code(s, k / 2);
    if k.is_multiple_of(2) { c.re } else { -c.im }
}
fn row(h: &Features, block: usize) -> [f64; 4] {
    if block == 0 {
        [h[0], h[1], h[2], h[3]]
    } else {
        [h[4], 1., 0., 0.]
    }
}
pub fn normalizer(h: &[Features]) -> f64 {
    h.iter()
        .map(|x| x.iter().map(|v| v.abs()).sum::<f64>() + 1.)
        .sum::<f64>()
        .max(1.)
}
pub fn features(h: &[Features]) -> Vec<[f64; 16]> {
    let n = normalizer(h);
    h.iter()
        .map(|h| {
            std::array::from_fn(|k| {
                row(h, k / 8)
                    .iter()
                    .enumerate()
                    .map(|(s, x)| x * source(k % 8, s))
                    .sum::<f64>()
                    / n
            })
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Head {
    pub coordinates: Vec<f64>,
    pub theta: Vec<Parameters>,
    pub coefficients: Vec<f64>,
}

impl Head {
    pub fn new(coordinates: Vec<f64>, theta: Vec<Parameters>) -> Result<Self> {
        data::coordinates(&coordinates)?;
        if !(1..=MAX_PROGRAMS).contains(&theta.len())
            || theta
                .iter()
                .flatten()
                .any(|x| !x.is_finite() || x.abs() > 1e12)
        {
            return Err("invalid finite complete-head parameters".into());
        }
        let mut coefficients = Vec::with_capacity(theta.len() * 16 * coordinates.len());
        for t in &theta {
            for k in 0..16 {
                for &e in &coordinates {
                    let values = if k < 8 {
                        [2. * e * t[0], 2. * e * t[1], 2. * e * t[2], 2. * e * t[3]]
                    } else {
                        [2. * e * t[4], -e * e * t[5], 0., 0.]
                    };
                    coefficients.push(
                        values
                            .iter()
                            .enumerate()
                            .map(|(s, v)| v * source(k % 8, s))
                            .sum(),
                    );
                }
            }
        }
        Ok(Self {
            coordinates,
            theta,
            coefficients,
        })
    }

    pub fn validate(&self) -> Result<()> {
        if *self != Self::new(self.coordinates.clone(), self.theta.clone())? {
            return Err("saved complete tensor differs from affine parameters".into());
        }
        Ok(())
    }

    /// Prediction evaluates every saved real/imaginary coefficient and raw class.
    pub fn logits(&self, h: &[Features]) -> Result<Vec<f64>> {
        data::features(h, self.theta.len())?;
        let phi = features(h);
        let v = self.coordinates.len();
        let mut z = vec![0.; v];
        for (p, phi) in phi.iter().enumerate() {
            for (k, x) in phi.iter().enumerate() {
                for (c, out) in z.iter_mut().enumerate() {
                    *out += self.coefficients[(p * 16 + k) * v + c] * x;
                }
            }
        }
        Ok(z)
    }

    /// Independent complex multiplication of the same complete saved tensor.
    pub fn independent_logits(&self, h: &[Features]) -> Result<Vec<f64>> {
        data::features(h, self.theta.len())?;
        let v = self.coordinates.len();
        let n = normalizer(h);
        Ok((0..v)
            .map(|c| {
                let mut out = Complex64::new(0., 0.);
                for (p, h) in h.iter().enumerate() {
                    for b in 0..2 {
                        for f in 0..4 {
                            let u: Complex64 = row(h, b)
                                .iter()
                                .enumerate()
                                .map(|(s, x)| code(s, f) * x / n)
                                .sum();
                            let k = (p * 16 + b * 8 + 2 * f) * v + c;
                            out +=
                                u * Complex64::new(self.coefficients[k], self.coefficients[k + v]);
                        }
                    }
                }
                out.re
            })
            .collect())
    }

    pub fn max_column_charge(&self) -> f64 {
        let v = self.coordinates.len();
        (0..v)
            .map(|c| {
                self.coefficients
                    .iter()
                    .skip(c)
                    .step_by(v)
                    .map(|x| x.abs())
                    .sum::<f64>()
            })
            .fold(0., f64::max)
    }
}

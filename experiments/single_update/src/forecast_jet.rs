//! Task #188: old value, directional tangent and a ball around their sum.
//! FORECAST_PROTOCOL.md fixes these finite propagation rules before measurement.
use crate::{algebra::linear, gradient::energy};

pub const EPS: f64 = f32::EPSILON as f64;
pub fn norm(x: &[f64]) -> f64 {
    energy(x).sqrt()
}
pub struct Jet {
    pub a: Vec<f64>,
    pub v: Vec<f64>,
    pub r: Vec<f64>,
    pub width: usize,
}
impl Jet {
    pub fn rows(&self) -> usize {
        self.r.len()
    }
    pub fn split(&self, offset: usize, width: usize) -> Self {
        let part = |x: &[f64]| {
            x.chunks_exact(self.width)
                .flat_map(|row| row[offset..offset + width].iter().copied())
                .collect()
        };
        Self {
            a: part(&self.a),
            v: part(&self.v),
            r: self.r.clone(),
            width,
        }
    }
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.width, other.width);
        let add = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x + y).collect();
        Self {
            a: add(&self.a, &other.a),
            v: add(&self.v, &other.v),
            r: add(&self.r, &other.r),
            width: self.width,
        }
    }
    pub fn dense(&self, old: &[f64], step: &[f64], output: usize) -> Self {
        let a = linear(&self.a, self.rows(), self.width, old, output);
        let mut v = linear(&self.v, self.rows(), self.width, old, output);
        let parameter = linear(&self.a, self.rows(), self.width, step, output);
        let mixed = linear(&self.v, self.rows(), self.width, step, output);
        for (v, p) in v.iter_mut().zip(parameter) {
            *v += p;
        }
        let matrix_bound = old
            .iter()
            .zip(step)
            .map(|(a, d)| (a + d).powi(2))
            .sum::<f64>()
            .sqrt();
        let r = self
            .r
            .iter()
            .zip(mixed.chunks_exact(output))
            .map(|(r, m)| matrix_bound * r + norm(m))
            .collect();
        Self {
            a,
            v,
            r,
            width: output,
        }
    }
    pub fn rms(&self) -> Self {
        let mut out = Self {
            a: vec![],
            v: vec![],
            r: vec![],
            width: self.width,
        };
        for (q, (a, v)) in self
            .a
            .chunks_exact(self.width)
            .zip(self.v.chunks_exact(self.width))
            .enumerate()
        {
            let n = self.width as f64;
            let s = (energy(a) / n + EPS).sqrt();
            let nominal: Vec<_> = a.iter().zip(v).map(|(a, v)| a + v).collect();
            let sn = (energy(&nominal) / n + EPS).sqrt();
            let av = a.iter().zip(v).map(|(a, v)| a * v).sum::<f64>();
            let mut defect = 0.;
            for k in 0..self.width {
                let old = a[k] / s;
                let tangent = v[k] / s - a[k] * av / (n * s.powi(3));
                defect += (nominal[k] / sn - old - tangent).powi(2);
                out.a.push(old);
                out.v.push(tangent);
            }
            out.r.push(2. / sn * self.r[q] + defect.sqrt());
        }
        out
    }
    pub fn sigmoid(&self) -> Self {
        let sigmoid = |x: f64| 1. / (1. + (-x).exp());
        let mut out = Self {
            a: vec![],
            v: vec![],
            r: vec![],
            width: self.width,
        };
        for (q, (a, v)) in self
            .a
            .chunks_exact(self.width)
            .zip(self.v.chunks_exact(self.width))
            .enumerate()
        {
            let mut defect = 0.;
            for k in 0..self.width {
                let old = sigmoid(a[k]);
                let tangent = old * (1. - old) * v[k];
                defect += (sigmoid(a[k] + v[k]) - old - tangent).powi(2);
                out.a.push(old);
                out.v.push(tangent);
            }
            out.r.push(self.r[q] / 4. + defect.sqrt());
        }
        out
    }
    pub fn radial(&self) -> Self {
        let h = self.width / 2;
        let mut out = Self {
            a: vec![],
            v: vec![],
            r: vec![],
            width: h,
        };
        for (q, (a, v)) in self
            .a
            .chunks_exact(self.width)
            .zip(self.v.chunks_exact(self.width))
            .enumerate()
        {
            let mut defect = 0.;
            for k in 0..h {
                let old = (a[k].powi(2) + a[k + h].powi(2) + EPS).sqrt();
                let tangent = (a[k] * v[k] + a[k + h] * v[k + h]) / old;
                let nominal = ((a[k] + v[k]).powi(2) + (a[k + h] + v[k + h]).powi(2) + EPS).sqrt();
                defect += (nominal - old - tangent).powi(2);
                out.a.push(old);
                out.v.push(tangent);
            }
            out.r.push(self.r[q] + defect.sqrt());
        }
        out
    }
}

#[derive(Clone, Copy)]
pub enum Product {
    Real,
    Gate,
    Diagonal,
    Complex,
}
impl Product {
    fn apply(self, a: &[f64], b: &[f64]) -> Vec<f64> {
        let h = a.len() / 2;
        match self {
            Self::Real | Self::Diagonal => a.iter().zip(b).map(|(x, y)| x * y).collect(),
            Self::Gate => a.iter().enumerate().map(|(i, x)| x * b[i % h]).collect(),
            Self::Complex => (0..2 * h)
                .map(|i| {
                    if i < h {
                        (a[i] * b[i] - a[i + h] * b[i + h]) / 2f64.sqrt()
                    } else {
                        (a[i - h] * b[i] + a[i] * b[i - h]) / 2f64.sqrt()
                    }
                })
                .collect(),
        }
    }
    fn max_block(a: &[f64], complex: bool) -> f64 {
        if complex {
            let h = a.len() / 2;
            (0..h).map(|i| a[i].hypot(a[i + h])).fold(0., f64::max)
        } else {
            a.iter().map(|x| x.abs()).fold(0., f64::max)
        }
    }
    pub fn jets(self, a: &Jet, b: &Jet) -> Jet {
        assert_eq!(a.rows(), b.rows());
        let mut out = Jet {
            a: vec![],
            v: vec![],
            r: vec![],
            width: a.width,
        };
        let scale = if matches!(self, Self::Complex) {
            1. / 2f64.sqrt()
        } else {
            1.
        };
        for q in 0..a.rows() {
            let av = &a.a[q * a.width..(q + 1) * a.width];
            let ad = &a.v[q * a.width..(q + 1) * a.width];
            let bv = &b.a[q * b.width..(q + 1) * b.width];
            let bd = &b.v[q * b.width..(q + 1) * b.width];
            let old = self.apply(av, bv);
            let left = self.apply(ad, bv);
            let right = self.apply(av, bd);
            let mixed = self.apply(ad, bd);
            let an: Vec<_> = av.iter().zip(ad).map(|(a, v)| a + v).collect();
            let bn: Vec<_> = bv.iter().zip(bd).map(|(a, v)| a + v).collect();
            let am = Self::max_block(&an, !matches!(self, Self::Real));
            let bm = Self::max_block(&bn, matches!(self, Self::Complex | Self::Diagonal));
            out.a.extend(old);
            out.v.extend(left.iter().zip(right).map(|(a, b)| a + b));
            out.r
                .push(norm(&mixed) + scale * (am * b.r[q] + bm * a.r[q] + a.r[q] * b.r[q]));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cancelled_product_tangent_retains_the_nonzero_mixed_term() {
        let a = Jet {
            a: vec![1.],
            v: vec![1.],
            r: vec![0.],
            width: 1,
        };
        let b = Jet {
            a: vec![1.],
            v: vec![-1.],
            r: vec![0.],
            width: 1,
        };
        let out = Product::Real.jets(&a, &b);
        assert_eq!(out.a, [1.]);
        assert_eq!(out.v, [0.]);
        assert_eq!(out.r, [1.]);
        assert_eq!((1. + 1.) * (1. - 1.) - out.a[0] - out.v[0], -1.);
    }
    #[test]
    fn normalization_transports_an_unknown_nonzero_input_error_near_zero() {
        let old = Jet {
            a: vec![0., 0.],
            v: vec![0.0001, -0.0002],
            r: vec![0.0003],
            width: 2,
        };
        let forecast = old.rms();
        let actual = Jet {
            a: vec![0.0003, -0.0004],
            v: vec![0.; 2],
            r: vec![0.],
            width: 2,
        }
        .rms();
        let error: Vec<_> = (0..2)
            .map(|i| actual.a[i] - forecast.a[i] - forecast.v[i])
            .collect();
        assert!(norm(&error) <= forecast.r[0]);
        assert!(forecast.r[0].is_finite() && forecast.r[0] > 0.);
    }
}

//! Task #188 Jet at 4b8d18e9 supplies unchanged values/tangents/legacy radii.
//! FT1/FT2 alter only the three predeclared ablation radii.
use crate::{
    algebra::linear,
    forecast_jet::{EPS, Jet, norm},
    matrix::Bound,
};
pub struct Tight {
    pub base: Jet,
    pub radii: [Vec<f64>; 4],
    pub defects: Vec<f64>,
    pub gains: [Vec<f64>; 4],
}
pub fn rms_gain(length: f64, radius: f64, width: usize, local: bool) -> f64 {
    let scale = (width as f64).sqrt();
    let legacy = 2. * scale / length;
    if local && radius < length {
        legacy.min(scale / (length * (length - radius)).sqrt())
    } else {
        legacy
    }
}
impl Tight {
    pub fn initial(base: Jet) -> Self {
        let n = base.rows();
        Self {
            radii: std::array::from_fn(|_| base.r.clone()),
            base,
            defects: vec![0.; n],
            gains: std::array::from_fn(|_| vec![1.; n]),
        }
    }
    pub fn split(&self, offset: usize, width: usize) -> Self {
        Self {
            base: self.base.split(offset, width),
            radii: self.radii.clone(),
            defects: vec![0.; self.base.rows()],
            gains: std::array::from_fn(|_| vec![1.; self.base.rows()]),
        }
    }
    pub fn add(&self, other: &Self) -> Self {
        let mut out = Self::initial(self.base.add(&other.base));
        for m in 1..4 {
            for q in 0..self.base.rows() {
                out.radii[m][q] = self.radii[m][q] + other.radii[m][q];
            }
        }
        out
    }
    pub fn dense(&self, old: &[f64], step: &[f64], output: usize, factor: Bound) -> Self {
        let mut out = Self::initial(self.base.dense(old, step, output));
        let mixed = linear(
            &self.base.v,
            self.base.rows(),
            self.base.width,
            step,
            output,
        );
        let legacy = old
            .iter()
            .zip(step)
            .map(|(a, d)| (a + d).powi(2))
            .sum::<f64>()
            .sqrt();
        for (q, row) in mixed.chunks_exact(output).enumerate() {
            out.defects[q] = norm(row);
            for m in 0..4 {
                let gain = if m == 1 || m == 3 {
                    factor.factor
                } else {
                    legacy
                };
                out.gains[m][q] = gain;
                if m > 0 {
                    out.radii[m][q] = gain * self.radii[m][q] + out.defects[q];
                }
            }
        }
        out
    }
    pub fn rms(&self) -> Self {
        let mut out = Self::initial(self.base.rms());
        let w = self.base.width;
        for q in 0..self.base.rows() {
            let nominal: Vec<_> = (0..w)
                .map(|k| self.base.a[q * w + k] + self.base.v[q * w + k])
                .collect();
            let energy = nominal.iter().map(|x| x * x).sum::<f64>();
            let length = (energy + w as f64 * EPS).sqrt();
            // Match the source defect's arithmetic order in every ablation.
            let s = (energy / w as f64 + EPS).sqrt();
            out.defects[q] = (0..w)
                .map(|k| (nominal[k] / s - out.base.a[q * w + k] - out.base.v[q * w + k]).powi(2))
                .sum::<f64>()
                .sqrt();
            for m in 0..4 {
                let gain = if m >= 2 {
                    rms_gain(length, self.radii[m][q], w, true).min(2. / s)
                } else {
                    2. / s
                };
                out.gains[m][q] = gain;
                if m > 0 {
                    out.radii[m][q] = gain * self.radii[m][q] + out.defects[q];
                }
            }
        }
        out
    }
    pub fn sigmoid(&self) -> Self {
        let mut out = Self::initial(self.base.sigmoid());
        let w = self.base.width;
        for q in 0..self.base.rows() {
            out.defects[q] = (0..w)
                .map(|k| {
                    let at = q * w + k;
                    let nominal = 1. / (1. + (-(self.base.a[at] + self.base.v[at])).exp());
                    (nominal - out.base.a[at] - out.base.v[at]).powi(2)
                })
                .sum::<f64>()
                .sqrt();
            for m in 0..4 {
                out.gains[m][q] = 0.25;
                if m > 0 {
                    out.radii[m][q] = self.radii[m][q] / 4. + out.defects[q];
                }
            }
        }
        out
    }
    pub fn radial(&self) -> Self {
        let mut out = Self::initial(self.base.radial());
        let w = self.base.width;
        let h = w / 2;
        for q in 0..self.base.rows() {
            out.defects[q] = (0..h)
                .map(|k| {
                    let at = q * w + k;
                    let nominal = ((self.base.a[at] + self.base.v[at]).powi(2)
                        + (self.base.a[at + h] + self.base.v[at + h]).powi(2)
                        + EPS)
                        .sqrt();
                    (nominal - out.base.a[q * h + k] - out.base.v[q * h + k]).powi(2)
                })
                .sum::<f64>()
                .sqrt();
            for m in 1..4 {
                out.radii[m][q] = self.radii[m][q] + out.defects[q];
            }
        }
        out
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn local_rms_uses_the_ball_condition_and_retains_the_global_fallback() {
        assert_eq!(rms_gain(2., 2., 4, true), 2.);
        assert_eq!(rms_gain(2., 3., 4, true), 2.);
        assert_eq!(rms_gain(2., 0., 4, true), 1.);
        assert!(rms_gain(2., 0.01, 4, true) < 1.01);
        let value = Tight::initial(Jet {
            a: vec![0., 0.],
            v: vec![1e-4, -2e-4],
            r: vec![3e-4],
            width: 2,
        });
        let out = value.rms();
        let actual = Jet {
            a: vec![3e-4, -4e-4],
            v: vec![0.; 2],
            r: vec![0.],
            width: 2,
        }
        .rms();
        let error = norm(
            &(0..2)
                .map(|k| actual.a[k] - out.base.a[k] - out.base.v[k])
                .collect::<Vec<_>>(),
        );
        assert!(out.radii.iter().all(|r| error <= r[0]));
    }
}

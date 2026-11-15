//! Actual eight-coordinate CM56 head, Head.lean at b5f1c9d7. This is the
//! complete realification of a fixed four-point Fourier code for every class.
//! Prediction uses expanded coefficients, not a nearest-coordinate shortcut.
use crate::{coordinates::Fit, *};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Head {
    pub fit: Fit,
    pub margin: f64,
    pub radius: f64,
    pub coefficients: Vec<f64>,
}

pub fn code(row: usize, column: usize) -> C {
    match (row * column) % 4 {
        0 => C::new(0.5, 0.),
        1 => C::new(0., 0.5),
        2 => C::new(-0.5, 0.),
        _ => C::new(0., -0.5),
    }
}

fn source(k: usize, state: usize) -> f64 {
    let z = code(state, k / 2);
    if k.is_multiple_of(2) { z.re } else { -z.im }
}

pub fn features(x: f64) -> [f64; 8] {
    let normalizer = (x.abs() + 1.).max(1.);
    std::array::from_fn(|k| (x * source(k, 0) + source(k, 1)) / normalizer)
}

impl Head {
    pub fn new(fit: Fit, margin: f64, radius: f64) -> Result<Self> {
        coordinates::validate_coordinates(&fit.coordinates)?;
        let mass = margin * fit.cost_per_margin();
        if !margin.is_finite()
            || margin <= 0.
            || !radius.is_finite()
            || radius <= 0.
            || !mass.is_finite()
            || mass > radius + 64. * f64::EPSILON * radius.max(1.)
            || !fit.slope.is_finite()
            || !fit.intercept.is_finite()
        {
            return Err("invalid margin or complete weighted coefficient budget".into());
        }
        let theta = [margin * fit.slope, margin * fit.intercept, margin];
        let mut coefficients = Vec::with_capacity(8 * fit.coordinates.len());
        for k in 0..8 {
            for &e in &fit.coordinates {
                coefficients.push(
                    2. * e * source(k, 0) * theta[0] + 2. * e * source(k, 1) * theta[1]
                        - e * e * source(k, 1) * theta[2],
                );
            }
        }
        let head = Self {
            fit,
            margin,
            radius,
            coefficients,
        };
        if head.max_column_l1() > radius + 64. * f64::EPSILON * radius.max(1.) {
            return Err("expanded complete head exceeds its radius".into());
        }
        Ok(head)
    }

    pub fn validate(&self) -> Result<()> {
        if *self != Self::new(self.fit.clone(), self.margin, self.radius)? {
            return Err("complete class-coordinate coefficients changed".into());
        }
        Ok(())
    }

    pub fn logits(&self, x: f64) -> Vec<f64> {
        assert!(x.is_finite() && x.abs() <= 4096.);
        let phi = features(x);
        let mut z = vec![0.; self.fit.coordinates.len()];
        for (k, feature) in phi.iter().enumerate() {
            for (c, out) in z.iter_mut().enumerate() {
                *out += self.coefficients[k * self.fit.coordinates.len() + c] * feature;
            }
        }
        z
    }

    /// Independent complex multiplication, using the saved complete tensor.
    pub fn independent_logits(&self, x: f64) -> Vec<f64> {
        let n = self.fit.coordinates.len();
        (0..n)
            .map(|c| {
                let mut z = C::new(0., 0.);
                for f in 0..4 {
                    let u = (code(0, f) * x + code(1, f)) / (x.abs() + 1.).max(1.);
                    let h = C::new(
                        self.coefficients[2 * f * n + c],
                        self.coefficients[(2 * f + 1) * n + c],
                    );
                    z += u * h;
                }
                z.re
            })
            .collect()
    }

    pub fn max_column_l1(&self) -> f64 {
        let n = self.fit.coordinates.len();
        (0..n)
            .map(|c| {
                self.coefficients
                    .iter()
                    .skip(c)
                    .step_by(n)
                    .map(|x| x.abs())
                    .sum::<f64>()
            })
            .fold(0., f64::max)
    }
}

//! CFt5/CM52 canonical copy tensor. State indices name vocabulary symbols;
//! Fourier phases are a basis, never an arithmetic answer rule.
use crate::*;
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Head {
    pub vocabulary: usize,
    pub code: Vec<C>,
    pub coefficients: Vec<f64>,
    pub radius: f64,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Audit {
    pub singular_min: f64,
    pub singular_max: f64,
    pub max_inversion_error: f64,
    pub max_column_l1: f64,
    pub scalar_work_forecast: u64,
    pub exact_real_certificate: bool,
}

impl Head {
    pub fn new(n: usize) -> Result<Self> {
        if !(2..=1024).contains(&n) {
            return Err("invalid copy-head vocabulary".into());
        }
        let code: Vec<_> = (0..n)
            .flat_map(|s| {
                (0..n).map(move |f| {
                    C::from_polar(
                        1. / (n as f64).sqrt(),
                        std::f64::consts::TAU * ((s * f) % n) as f64 / n as f64,
                    )
                })
            })
            .collect();
        let mut coefficients = vec![0.; 2 * n * n];
        for state in 0..n {
            for f in 0..n {
                let h = code[state * n + f].conj() * 24.;
                coefficients[(2 * f) * n + state] += h.re;
                coefficients[(2 * f + 1) * n + state] += h.im;
            }
        }
        Ok(Self {
            vocabulary: n,
            code,
            coefficients,
            radius: 64. * n as f64,
        })
    }
    pub fn validate(&self) -> Result<()> {
        if *self != Self::new(self.vocabulary)? {
            return Err("canonical full complex code/head changed".into());
        }
        if self.max_column_l1() > self.radius + 1e-8 {
            return Err("complete copy-head column budget exceeded".into());
        }
        Ok(())
    }
    pub fn max_column_l1(&self) -> f64 {
        (0..self.vocabulary)
            .map(|c| {
                self.coefficients
                    .iter()
                    .skip(c)
                    .step_by(self.vocabulary)
                    .map(|v| v.abs())
                    .sum::<f64>()
            })
            .fold(0., f64::max)
    }
    pub fn logits(&self, phi: &[C]) -> Vec<f64> {
        let n = self.vocabulary;
        assert_eq!(phi.len(), n);
        let mut z = vec![0.; n];
        for (f, &value) in phi.iter().enumerate() {
            for (c, out) in z.iter_mut().enumerate() {
                *out += self.coefficients[(2 * f) * n + c] * value.re
                    - self.coefficients[(2 * f + 1) * n + c] * value.im;
            }
        }
        z
    }
    pub fn state_logits(&self, state: usize) -> Vec<f64> {
        let n = self.vocabulary;
        self.logits(&self.code[state * n..(state + 1) * n])
    }
    /// Independent class-major complex multiplication; no ideal one-hot
    /// output substitutes for the actual retained tensor.
    pub fn independent_table(&self) -> Vec<Vec<f64>> {
        let n = self.vocabulary;
        (0..n)
            .map(|state| {
                (0..n)
                    .map(|c| {
                        (0..n)
                            .map(|f| {
                                (C::new(
                                    self.coefficients[(2 * f) * n + c],
                                    self.coefficients[(2 * f + 1) * n + c],
                                ) * self.code[state * n + f])
                                    .re
                            })
                            .sum()
                    })
                    .collect()
            })
            .collect()
    }
    pub fn audit(&self) -> Result<Audit> {
        self.validate()?;
        let n = self.vocabulary;
        let work = 96 * (n as u64).pow(3);
        if work > 2_000_000_000 {
            return Err("copy-head audit exceeds2B scalar-work forecast".into());
        }
        let singular = linalg::svd(&self.code, n)?.values;
        if singular.len() != n
            || singular
                .iter()
                .any(|s| !s.is_finite() || (s - 1.).abs() > 1e-8)
        {
            return Err("vocabulary code failed numerical unitary audit".into());
        }
        let table = self.independent_table();
        let mut error = 0.0f64;
        for (s, row) in table.iter().enumerate() {
            let actual = self.state_logits(s);
            for (c, &z) in row.iter().enumerate() {
                if (z - actual[c]).abs() > 1e-10 {
                    return Err("copy head realification differs".into());
                }
                error = error.max((z - if c == s { 24. } else { 0. }).abs());
            }
        }
        if error > 1e-8 {
            return Err("full copy-head inversion failed".into());
        }
        Ok(Audit {
            singular_min: singular[n - 1],
            singular_max: singular[0],
            max_inversion_error: error,
            max_column_l1: self.max_column_l1(),
            scalar_work_forecast: work,
            exact_real_certificate: false,
        })
    }
}

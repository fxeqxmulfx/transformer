//! CM53 complete conjugate decoder blocks, ported from3badaa30. Templates
//! freeze after training-only discovery. Unknown raw-symbol states use an
//! explicit identity-copy completion; other unknown states use raw0.
//! ofDecoders_mem permits this consistent completion, but CM53/CM54 do not
//! identify its unknown rows. The partial dictionaries remain None there.
use crate::*;
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Head {
    pub vocabulary: usize,
    pub code: Vec<C>,
    pub decoders: Vec<Vec<Option<usize>>>,
    pub coefficients: Vec<Vec<f64>>,
    pub radius: f64,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Audit {
    pub singular_min: f64,
    pub singular_max: f64,
    pub max_inversion_error: f64,
    pub max_column_l1: f64,
    pub real_head_coordinates: usize,
    pub scalar_work_forecast: u64,
    pub exact_real_certificate: bool,
}
impl Head {
    pub fn new(vocabulary: usize, decoders: Vec<Vec<Option<usize>>>) -> Result<Self> {
        if !(2..=1024).contains(&vocabulary)
            || decoders.is_empty()
            || decoders.len() > 128
            || decoders
                .iter()
                .any(|d| d.len() != STATES || d.iter().flatten().any(|&c| c >= vocabulary))
        {
            return Err("invalid complete latent decoder dimensions".into());
        }
        let code = copy::head::Head::new(STATES)?.code;
        let coefficients = decoders
            .iter()
            .map(|decoder| {
                let mut head = vec![0.; 2 * STATES * vocabulary];
                for (state, label) in decoder.iter().enumerate() {
                    let c = label.unwrap_or(if state < vocabulary { state } else { 0 });
                    for f in 0..STATES {
                        let h = code[state * STATES + f].conj() * 24.;
                        head[2 * f * vocabulary + c] += h.re;
                        head[(2 * f + 1) * vocabulary + c] += h.im;
                    }
                }
                head
            })
            .collect();
        Ok(Self {
            vocabulary,
            code,
            decoders,
            coefficients,
            radius: 64. * STATES as f64,
        })
    }
    pub fn validate(&self) -> Result<()> {
        if *self != Self::new(self.vocabulary, self.decoders.clone())? {
            return Err("complete learned decoder code or coefficients changed".into());
        }
        Ok(())
    }
    pub fn block_logits(&self, j: usize, state: usize) -> Vec<f64> {
        let mut z = vec![0.; self.vocabulary];
        for f in 0..STATES {
            let value = self.code[state * STATES + f];
            for (c, v) in z.iter_mut().enumerate() {
                *v += self.coefficients[j][2 * f * self.vocabulary + c] * value.re
                    - self.coefficients[j][(2 * f + 1) * self.vocabulary + c] * value.im;
            }
        }
        z
    }
    pub fn logits(&self, states: &[usize], weights: &[f64]) -> Vec<f64> {
        assert_eq!(states.len(), self.coefficients.len());
        assert_eq!(states.len(), weights.len());
        let mut z = vec![0.; self.vocabulary];
        for (j, (&state, &w)) in states.iter().zip(weights).enumerate() {
            // These are actual affine joint-head coordinates. No ideal
            // dictionary logit replaces any real or imaginary product.
            for f in 0..STATES {
                let value = self.code[state * STATES + f];
                for (c, v) in z.iter_mut().enumerate() {
                    *v += (w * self.coefficients[j][2 * f * self.vocabulary + c]) * value.re
                        - (w * self.coefficients[j][(2 * f + 1) * self.vocabulary + c]) * value.im;
                }
            }
        }
        z
    }
    pub fn independent_table(&self) -> Vec<Vec<Vec<f64>>> {
        self.coefficients
            .iter()
            .map(|block| {
                (0..STATES)
                    .map(|state| {
                        (0..self.vocabulary)
                            .map(|c| {
                                (0..STATES)
                                    .map(|f| {
                                        (C::new(
                                            block[2 * f * self.vocabulary + c],
                                            block[(2 * f + 1) * self.vocabulary + c],
                                        ) * self.code[state * STATES + f])
                                            .re
                                    })
                                    .sum()
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }
    pub fn max_column_l1(&self, weights: &[f64]) -> f64 {
        (0..self.vocabulary)
            .map(|c| {
                self.coefficients
                    .iter()
                    .zip(weights)
                    .map(|(h, &w)| {
                        h.iter()
                            .skip(c)
                            .step_by(self.vocabulary)
                            .map(|v| (w * v).abs())
                            .sum::<f64>()
                    })
                    .sum()
            })
            .fold(0., f64::max)
    }
    pub fn audit(&self, weights: &[f64]) -> Result<Audit> {
        self.validate()?;
        let singular = linalg::svd(&self.code, STATES)?.values;
        if singular.len() != STATES
            || singular
                .iter()
                .any(|v| !v.is_finite() || (v - 1.).abs() > 1e-8)
        {
            return Err("latent source code failed numerical unitary audit".into());
        }
        let table = self.independent_table();
        let mut error = 0.0f64;
        for (j, block) in table.iter().enumerate() {
            for (s, row) in block.iter().enumerate() {
                let actual = self.block_logits(j, s);
                for (c, &v) in row.iter().enumerate() {
                    if (v - actual[c]).abs() > 1e-10 {
                        return Err("latent realification differs".into());
                    }
                    error = error.max(
                        (v - if c
                            == self.decoders[j][s].unwrap_or(if s < self.vocabulary {
                                s
                            } else {
                                0
                            }) {
                            24.
                        } else {
                            0.
                        })
                        .abs(),
                    );
                }
            }
        }
        let budget = self.max_column_l1(weights);
        if error > 1e-8 || budget > self.radius + 1e-8 {
            return Err("complete latent inversion/budget fails".into());
        }
        Ok(Audit {
            singular_min: singular[STATES - 1],
            singular_max: singular[0],
            max_inversion_error: error,
            max_column_l1: budget,
            real_head_coordinates: self.coefficients.len() * 2 * STATES * self.vocabulary,
            scalar_work_forecast: (96 * STATES.pow(3)
                + 32 * self.coefficients.len() * STATES.pow(2) * self.vocabulary)
                as u64,
            exact_real_certificate: false,
        })
    }
}

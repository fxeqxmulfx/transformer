use super::*;
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Audit {
    pub raw_latent_operator: Vec<f64>,
    pub raw_latent_nuclear: Vec<f64>,
    pub head_column_l1: Vec<f64>,
    pub train_ce: f64,
    pub raw_training_errors: usize,
    pub known_cells: usize,
    pub svd_matrices: usize,
    pub complex_entries_audited: usize,
    pub head_dot_product_terms: usize,
    pub buffer_preflight_bytes: u64,
    pub spectral_bounds_apply_to: String,
    pub exact_real_arithmetic_certificate: bool,
}
impl Model {
    pub fn audit(&self, samples: &[[usize; 2]]) -> Result<Audit> {
        self.observations(samples)?;
        let (mut operator, mut nuclear) = (vec![], vec![]);
        for f in 0..self.q {
            let actual: Vec<_> = self
                .latent
                .iter()
                .map(|&z| {
                    C::from_polar(
                        1.,
                        std::f64::consts::TAU * (f * z as usize) as f64 / self.q as f64,
                    )
                })
                .collect();
            let singular = linalg::svd(&actual, self.n)?.values;
            let sum: f64 = singular.iter().sum();
            if singular.iter().any(|x| !x.is_finite() || *x < 0.)
                || singular[0] > self.n as f64 + 1e-7
                || sum > self.n as f64 + 1e-7
            {
                return Err("actual latent Fourier matrix exceeds radius/nuclear budget N".into());
            }
            operator.push(singular[0]);
            nuclear.push(sum);
        }
        let mut ce = 0.;
        let mut errors = 0;
        for &[cell, label] in samples {
            // Recompute headLogits as a feature dot product, independently of
            // the predictor's direct slice. All original classes enter the CE.
            let z = (self.rows[cell % self.n] as i128 + self.columns[cell / self.n] as i128)
                .rem_euclid(self.q as i128) as usize;
            let logits: Vec<f64> = (0..self.vocabulary)
                .map(|c| {
                    (0..self.q)
                        .map(|k| self.coefficients[k * self.vocabulary + c] * f64::from(k == z))
                        .sum()
                })
                .collect();
            let top = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            ce += logits.iter().map(|x| (x - top).exp()).sum::<f64>().ln() + top - logits[label];
            let prediction = (1..self.vocabulary)
                .fold(0, |best, c| if logits[c] > logits[best] { c } else { best });
            errors += usize::from(prediction != label);
            if logits != self.logits(cell) {
                return Err("actual head dot product differs".into());
            }
        }
        Ok(Audit {
            raw_latent_operator: operator,
            raw_latent_nuclear: nuclear,
            head_column_l1: (0..self.vocabulary)
                .map(|c| {
                    (0..self.q)
                        .map(|z| self.coefficients[z * self.vocabulary + c].abs())
                        .sum()
                })
                .collect(),
            train_ce: ce / samples.len() as f64,
            raw_training_errors: errors,
            known_cells: (0..self.n * self.n)
                .filter(|&cell| self.known(cell))
                .count(),
            svd_matrices: self.q,
            complex_entries_audited: self.q * self.n * self.n,
            head_dot_product_terms: samples.len() * self.q * self.vocabulary,
            buffer_preflight_bytes: memory_bound(self.n, self.q, self.vocabulary)?,
            spectral_bounds_apply_to: "latent_moments".into(),
            exact_real_arithmetic_certificate: false,
        })
    }
}

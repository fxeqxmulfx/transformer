use super::*;
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Audit {
    pub code_singular_min: Vec<f64>,
    pub code_singular_max: Vec<f64>,
    pub code_nuclear: Vec<f64>,
    pub head_column_l1: Vec<f64>,
    pub train_ce: f64,
    pub train_errors: usize,
    pub known_training_prefixes: usize,
    pub svd_matrices: usize,
    pub complex_entries: usize,
    pub scalar_work_forecast: u64,
    pub spectral_scope: String,
    pub exact_real_certificate: bool,
    pub unseen_identification: bool,
}
impl Model {
    pub fn code_audit_forecast(&self) -> Result<u64> {
        let work = self
            .blocks
            .iter()
            .map(|b| {
                let n = b.states() as u64;
                64 * n * n * n + 8 * n * n * self.vocabulary as u64
            })
            .sum::<u64>();
        if work > 2_000_000_000 {
            return Err("saved complex code audit work exceeds 2B forecast".into());
        }
        Ok(work)
    }
    pub fn audit_forecast(&self, samples: &[Encoded]) -> Result<u64> {
        let work = self.code_audit_forecast()?
            + samples
                .iter()
                .map(|s| 16 * self.blocks[s.block].states() as u64 * self.vocabulary as u64)
                .sum::<u64>();
        if work > 2_000_000_000 {
            return Err("original-data complex audit exceeds 2B scalar-work forecast".into());
        }
        Ok(work)
    }
    pub fn audit(&self, input: &Input) -> Result<Audit> {
        let encoded = self.observations(input)?;
        let forecast = self.audit_forecast(&encoded)?;
        let (mut low, mut high, mut nuclear) = (vec![], vec![], vec![]);
        for b in &self.blocks {
            let n = b.states();
            let singular = linalg::svd(&b.code, n)?.values;
            if singular.len() != n
                || singular
                    .iter()
                    .any(|x| !x.is_finite() || (x - 1.).abs() > 1e-7)
            {
                return Err(
                    "saved complex categorical code is not numerically flat/unitary".into(),
                );
            }
            low.push(singular[n - 1]);
            high.push(singular[0]);
            nuclear.push(singular.iter().sum());
        }
        let (mut ce, mut errors) = (0., 0);
        for s in &encoded {
            let b = &self.blocks[s.block];
            let n = b.states();
            let mut z = vec![0.; self.vocabulary];
            // Recompute in complex arithmetic, separately from logits_state's
            // realification. Both learned real and imaginary coordinates matter.
            for (c, value) in z.iter_mut().enumerate() {
                for f in 0..n {
                    let row = (b.first_row + 2 * f) * self.vocabulary;
                    *value += (C::new(
                        self.coefficients[row + c],
                        self.coefficients[row + self.vocabulary + c],
                    ) * b.code[s.state * n + f])
                        .re;
                }
            }
            if z.iter()
                .zip(self.logits_state(s.block, s.state))
                .any(|(a, b)| (a - b).abs() > 1e-10)
            {
                return Err("actual complex logits differ from the learned real tensor".into());
            }
            let top = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            ce += z.iter().map(|v| (v - top).exp()).sum::<f64>().ln() + (top - z[s.label]);
            let chosen = (1..z.len()).fold(0, |best, c| if z[c] > z[best] { c } else { best });
            errors += usize::from(chosen != s.label);
        }
        Ok(Audit {
            code_singular_min: low,
            code_singular_max: high,
            code_nuclear: nuclear,
            head_column_l1: (0..self.vocabulary)
                .map(|c| {
                    self.coefficients
                        .iter()
                        .skip(c)
                        .step_by(self.vocabulary)
                        .map(|x| x.abs())
                        .sum()
                })
                .collect(),
            train_ce: ce / encoded.len() as f64,
            train_errors: errors,
            known_training_prefixes: encoded.len(),
            svd_matrices: self.blocks.len(),
            complex_entries: self.blocks.iter().map(|b| b.code.len()).sum(),
            scalar_work_forecast: forecast,
            spectral_scope: "fixed categorical code bases, not the empirical target matrix".into(),
            exact_real_certificate: false,
            unseen_identification: false,
        })
    }
}

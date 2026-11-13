//! Ported refined_quotient/audit.rs at285dcdb6, with CM34 vector-range and
//! raw-label binding checks. Independent complex products and numerical Gram/SVD
//! remain audits of actual tensors, not exact-real floating-point proofs.
use crate::*;
impl Model {
    pub fn validate(&self) -> Result<()> {
        let bound = self.bounds()?;
        self.encoding.validate(self.n, self.vocabulary)?;
        let v = self.vocabulary;
        if self.blocks.len() != self.encoding.keys.len()
            || self.table.len() != bound.states * v
            || self.coefficients.len() != bound.real_coefficients
            || self.radius != 64. * bound.states as f64
            || self.table.iter().any(|x| !x.is_finite() || x.abs() > 32.)
            || self.coefficients.iter().any(|x| !x.is_finite())
        {
            return Err("invalid refined dimensions, finite table or radius".into());
        }
        let mut offset = 0;
        let mut seen = vec![false; self.n * self.n];
        for (z, b) in self.blocks.iter().enumerate() {
            let n = b.states();
            if b.offset != offset
                || b.default_label >= v
                || b.default_label != self.encoding.default_label(z)
                || b.code.len() != n * n
                || b.observed.windows(2).any(|s| s[0] >= s[1])
                || b.observed_labels.len() != b.observed.len()
                || b.observed_labels.iter().any(|&c| c >= v)
            {
                return Err("invalid refined block/dictionary shape".into());
            }
            for (&cell, &label) in b.observed.iter().zip(&b.observed_labels) {
                if cell >= seen.len()
                    || seen[cell]
                    || self.latent(cell) != Some(z)
                    || self.encoding.code.label(label) != Some(self.encoding.keys[z].as_slice())
                {
                    return Err("observed identity differs from its fixed latent fiber".into());
                }
                seen[cell] = true;
            }
            for s in 0..n {
                for f in 0..n {
                    if b.code[s * n + f] != model::phase(n, s, f) {
                        return Err(
                            "saved code differs from the fixed categorical Fourier block".into(),
                        );
                    }
                }
            }
            let d = (offset + n - 1) * v;
            let actual = self.logits_state(z, n - 1);
            for (c, &logit) in actual.iter().enumerate() {
                let expected = if c == b.default_label { 24. } else { 0. };
                if self.table[d + c] != expected || !close(logit, expected, 1e-7) {
                    return Err("actual default logits or affine table constraint changed".into());
                }
            }
            offset += n;
        }
        for c in 0..v {
            let sum: f64 = self
                .coefficients
                .iter()
                .skip(c)
                .step_by(v)
                .map(|x| x.abs())
                .sum();
            if sum > self.radius + 1e-8 {
                return Err("complex L1 column exceeds64N".into());
            }
        }
        Ok(())
    }
}
pub(crate) fn close(a: f64, b: f64, tolerance: f64) -> bool {
    a.is_finite() && b.is_finite() && (a - b).abs() <= tolerance
}
pub fn complex_logits(model: &Model, location: (usize, usize)) -> Vec<f64> {
    let b = &model.blocks[location.0];
    (0..model.vocabulary)
        .map(|c| {
            (0..b.states())
                .map(|f| {
                    let h = 2 * (b.offset + f) * model.vocabulary;
                    let weight = C::new(
                        model.coefficients[h + c],
                        model.coefficients[h + model.vocabulary + c],
                    );
                    model.blocks[location.0].code[location.1 * b.states() + f] * weight
                })
                .sum::<C>()
                .re
        })
        .collect()
}
pub(crate) fn transformed(model: &Model, table: &[f64]) -> Vec<f64> {
    let mut h = vec![0.; model.coefficients.len()];
    for b in &model.blocks {
        for f in 0..b.states() {
            for c in 0..model.vocabulary {
                let value: C = (0..b.states())
                    .map(|s| {
                        b.code[s * b.states() + f].conj()
                            * table[(b.offset + s) * model.vocabulary + c]
                    })
                    .sum();
                let row = 2 * (b.offset + f) * model.vocabulary;
                h[row + c] = value.re;
                h[row + model.vocabulary + c] = value.im;
            }
        }
    }
    h
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Audit {
    pub bounds: limits::Bounds,
    pub singular_min: Vec<f64>,
    pub singular_max: Vec<f64>,
    pub nuclear: Vec<f64>,
    pub gram_error: f64,
    pub table_logit_error: f64,
    pub coefficient_error: f64,
    pub column_l1: Vec<f64>,
    pub train_ce: f64,
    pub train_errors: usize,
    pub svd_matrices: usize,
    pub spectral_scope: String,
    pub exact_real_certificate: bool,
    pub unseen_identification: bool,
}
impl Model {
    pub fn audit(&self, samples: &[[usize; 2]]) -> Result<Audit> {
        self.validate()?;
        self.encode(samples)?;
        let bounds = self.bounds()?;
        let (mut low, mut high, mut nuclear) = (vec![], vec![], vec![]);
        let mut gram_error = 0_f64;
        let mut table_logit_error = 0_f64;
        for (z, b) in self.blocks.iter().enumerate() {
            let n = b.states();
            let singular = linalg::svd(&b.code, n)?.values;
            if singular.len() != n || singular.iter().any(|x| !close(*x, 1., 1e-7)) {
                return Err("actual block singular values are not numerically unitary".into());
            }
            low.push(singular[n - 1]);
            high.push(singular[0]);
            nuclear.push(singular.iter().sum());
            for i in 0..n {
                for j in 0..n {
                    let left: C = (0..n)
                        .map(|k| b.code[k * n + i].conj() * b.code[k * n + j])
                        .sum();
                    let right: C = (0..n)
                        .map(|k| b.code[i * n + k] * b.code[j * n + k].conj())
                        .sum();
                    let target = C::new(f64::from(i == j), 0.);
                    gram_error = gram_error
                        .max((left - target).norm())
                        .max((right - target).norm());
                }
                for (c, (&actual, &real)) in complex_logits(self, (z, i))
                    .iter()
                    .zip(self.logits_state(z, i).iter())
                    .enumerate()
                {
                    if !close(actual, real, 1e-9) {
                        return Err("complex and realified forward paths differ".into());
                    }
                    table_logit_error = table_logit_error
                        .max((actual - self.table[(b.offset + i) * self.vocabulary + c]).abs());
                }
            }
        }
        let coefficient_error = transformed(self, &self.table)
            .iter()
            .zip(&self.coefficients)
            .map(|(a, b)| (a - b).abs())
            .fold(0., f64::max);
        if gram_error > 1e-7 || table_logit_error > 1e-7 || coefficient_error > 1e-8 {
            return Err("actual Gram, table logits or conjugate coefficients fail replay".into());
        }
        let mut train_ce = 0.;
        let mut train_errors = 0;
        for &[cell, label] in samples {
            let z = complex_logits(self, self.location(cell));
            let top = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            train_ce += z.iter().map(|x| (x - top).exp()).sum::<f64>().ln() + top - z[label];
            let best = (1..z.len()).fold(0, |best, c| if z[c] > z[best] { c } else { best });
            train_errors += usize::from(best != label);
        }
        Ok(Audit {
            bounds,
            singular_min: low,
            singular_max: high,
            nuclear,
            gram_error,
            table_logit_error,
            coefficient_error,
            train_ce: train_ce / samples.len() as f64,
            train_errors,
            column_l1: (0..self.vocabulary)
                .map(|c| {
                    self.coefficients
                        .iter()
                        .skip(c)
                        .step_by(self.vocabulary)
                        .map(|x| x.abs())
                        .sum()
                })
                .collect(),
            svd_matrices: self.blocks.len(),
            spectral_scope: "fixed categorical block basis; not empirical target or learned head"
                .into(),
            exact_real_certificate: false,
            unseen_identification: false,
        })
    }
}

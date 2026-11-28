//! Character expectation of the complete softmax on the 96 by 96 unit block.
//! The target phase comes from the completion of the training observations.
use super::{N, eq};
use crate::{Result, division, svd};
use serde_json::{Value, json};

type Complex = [f64; 2];

pub(super) struct PhaseTable {
    codes: Vec<Complex>,
    learned: Vec<Complex>,
    residual: Vec<Complex>,
    minimum_gap: f64,
}

impl PhaseTable {
    pub(super) fn new() -> Self {
        let mut codes = vec![[0.0; 2]; division::shape().vocab];
        let mut residue = 1;
        for power in 0..N {
            let angle = std::f64::consts::TAU * power as f64 / N as f64;
            let (sine, cosine) = angle.sin_cos();
            codes[eq::NUM0 + residue] = [cosine, sine];
            residue = 5 * residue % eq::P;
        }
        Self {
            codes,
            learned: Vec::with_capacity(N * N),
            residual: Vec::with_capacity(N * N),
            minimum_gap: f64::INFINITY,
        }
    }

    /// Inputs arrive in row-major order after omitting the zero numerator.
    pub(super) fn observe(&mut self, logits: &[f32], target_residue: usize, gap: f64) {
        let max = f64::from(logits.iter().copied().fold(f32::NEG_INFINITY, f32::max));
        let mut mass = 0.0;
        let mut sum = [0.0; 2];
        for (&z, code) in logits.iter().zip(&self.codes) {
            let weight = (f64::from(z) - max).exp();
            mass += weight;
            sum[0] += weight * code[0];
            sum[1] += weight * code[1];
        }
        let entry = [sum[0] / mass, sum[1] / mass];
        let target = self.codes[eq::NUM0 + target_residue];
        self.learned.push(entry);
        self.residual
            .push([entry[0] - target[0], entry[1] - target[1]]);
        self.minimum_gap = self.minimum_gap.min(gap);
    }

    pub(super) fn report(self) -> Result<Value> {
        if self.learned.len() != N * N || !self.minimum_gap.is_finite() {
            return Err("expected all 9216 finite unit-row outputs".into());
        }
        let values = singular_values(&self.learned, N);
        let nuclear: f64 = values.iter().sum();
        let residual_nuclear: f64 = singular_values(&self.residual, N).iter().sum();
        let max_entry = self
            .residual
            .iter()
            .map(|z| z[0].hypot(z[1]))
            .fold(0.0, f64::max);
        // Every competitor has probability at most exp(-gap), and the
        // total mass on wrong tokens is at most one.
        let wrong_mass = ((self.codes.len() - 1) as f64 * (-self.minimum_gap).exp()).min(1.0);
        let bound = 2.0 * (N as f64).powf(1.5) * wrong_mass;
        Ok(json!({
            "unit_cells":N*N, "output_tokens":self.codes.len(), "generator":5,
            "target_nuclear_norm":N, "nuclear_norm":nuclear, "singular_values":values,
            "residual_nuclear_norm":residual_nuclear, "max_entry_error":max_entry,
            "minimum_unit_logit_gap":self.minimum_gap,
            "nuclear_error_bound_approximation":bound,
            "margin_at_least_19":self.minimum_gap>=19.0
        }))
    }
}

/// The real block [Re -Im; Im Re] has each complex singular value twice.
fn singular_values(matrix: &[Complex], size: usize) -> Vec<f64> {
    let width = 2 * size;
    let mut real = vec![0.0; width * width];
    for (k, &[re, im]) in matrix.iter().enumerate() {
        let base = (k / size) * width + k % size;
        real[base] = re;
        real[base + size] = -im;
        real[base + size * width] = im;
        real[base + size * width + size] = re;
    }
    svd::thin(&real, width, width)
        .values
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| (pair[0] + pair[1]) * 0.5)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complex_spectrum_and_complete_vocabulary_softmax() {
        // [1 i; -i 1] has singular values 2 and 0.
        let values = singular_values(&[[1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [1.0, 0.0]], 2);
        assert!((values[0] - 2.0).abs() < 1e-12 && values[1] < 1e-12);
        let mut table = PhaseTable::new();
        let mut z = vec![-1e4; table.codes.len()];
        z[eq::NUM0 + 1] = 1e4;
        z[eq::EQ] = 1e4;
        table.observe(&z, 1, 0.0);
        assert_eq!(table.learned[0], [0.5, 0.0]);
        assert_eq!(table.residual[0], [-0.5, 0.0]);
        assert!(table.report().is_err());
    }
}

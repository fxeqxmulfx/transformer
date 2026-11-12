//! Orthogonal projection onto observable affine readings. Dependent rows are
//! handled by a declared spectral cutoff; inconsistent rows cause a refusal.
use crate::{C, Result, matrix};

pub struct Affine {
    pub rows: Vec<Vec<f64>>,
    pub targets: Vec<f64>,
    inverse: Vec<Vec<f64>>,
    pub rank: usize,
    pub consistency: f64,
}

impl Affine {
    pub fn new(mut rows: Vec<Vec<f64>>, mut targets: Vec<f64>, cutoff: f64) -> Result<Self> {
        let n = rows.len();
        if n == 0
            || n > 194
            || targets.len() != n
            || rows[0].is_empty()
            || rows[0].len() > 64
            || !cutoff.is_finite()
            || !(1e-14..=1e-6).contains(&cutoff)
            || rows
                .iter()
                .any(|r| r.len() != rows[0].len() || r.iter().any(|v| !v.is_finite()))
            || targets.iter().any(|v| !v.is_finite())
        {
            return Err("invalid affine projector input".into());
        }
        for (row, target) in rows.iter_mut().zip(&mut targets) {
            let norm = matrix::dot(row, row).sqrt();
            let scale = norm.max(1e-14);
            for value in row {
                *value /= scale;
            }
            *target /= scale;
        }
        let mut gram = vec![C::default(); n * n];
        for j in 0..n {
            for i in 0..n {
                gram[i + n * j] = C::new(matrix::dot(&rows[i], &rows[j]), 0.);
            }
        }
        let e = matrix::eigen(&gram, n)?;
        let threshold = e.values[n - 1].max(1.) * cutoff;
        let rank = e.values.iter().filter(|&&v| v > threshold).count();
        let pinv = matrix::reconstruct(
            &e,
            &e.values
                .iter()
                .map(|&v| if v > threshold { 1. / v } else { 0. })
                .collect::<Vec<_>>(),
        );
        let inverse: Vec<_> = (0..n)
            .map(|i| (0..n).map(|j| pinv[i + n * j].re).collect())
            .collect();
        let mut out = Self {
            rows,
            targets,
            inverse,
            rank,
            consistency: 0.,
        };
        let zero = vec![0.; out.rows[0].len()];
        out.consistency = out.residual(&out.project(&zero).0);
        Ok(out)
    }

    /// Returns the projected coordinate and its row multipliers, so the
    /// displacement can be independently checked as A^T lambda.
    pub fn project(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let residual: Vec<_> = self
            .rows
            .iter()
            .zip(&self.targets)
            .map(|(row, b)| matrix::dot(row, x) - b)
            .collect();
        let multipliers: Vec<_> = self
            .inverse
            .iter()
            .map(|r| matrix::dot(r, &residual))
            .collect();
        let at = self.transpose(&multipliers);
        (x.iter().zip(at).map(|(&x, a)| x - a).collect(), multipliers)
    }

    pub fn transpose(&self, y: &[f64]) -> Vec<f64> {
        assert_eq!(y.len(), self.rows.len());
        let mut out = vec![0.; self.rows[0].len()];
        for (row, &v) in self.rows.iter().zip(y) {
            for (o, &a) in out.iter_mut().zip(row) {
                *o += v * a;
            }
        }
        out
    }
    pub fn residual(&self, x: &[f64]) -> f64 {
        self.rows
            .iter()
            .zip(&self.targets)
            .map(|(row, b)| (matrix::dot(row, x) - b).abs())
            .fold(0., f64::max)
    }
}

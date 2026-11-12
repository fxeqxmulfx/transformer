//! The Fourier moments of an even table on Z/q, q = 2 * orbits - 1.
//! For P(z) = P(-z) the moment M_f = sum_z P(z) exp(2 pi i f z / q) is real and
//! even in f: M_f = P(0) + 2 sum_{c>=1} P(c) cos(2 pi f c / q). So one real
//! orbits x orbits cosine matrix replaces the complex transform of length q,
//! and its inverse is P(c) = (M_0 + 2 sum_{f>=1} M_f cos(2 pi f c / q)) / q.
//! Lean `moment_reflectLatent` is the identity M_f(reflected P) = M_{-f}(P)
//! behind this; the code below is checked against direct complex sums in tests.
use crate::{Result, weight};

pub struct Cosine {
    orbits: usize,
    /// Row-major `[f][c]`: weight(c) * cos(2 pi f c / q).
    forward: Vec<f64>,
    /// Row-major `[c][f]`: weight(f) * cos(2 pi f c / q) / q.
    inverse: Vec<f64>,
}
impl Cosine {
    pub fn new(orbits: usize) -> Result<Self> {
        if !(2..=97).contains(&orbits) {
            return Err("latent orbits must lie in 2..=97".into());
        }
        let q = 2 * orbits - 1;
        let cosine =
            |f: usize, c: usize| (std::f64::consts::TAU * ((f * c) % q) as f64 / q as f64).cos();
        let mut forward = vec![0.; orbits * orbits];
        let mut inverse = vec![0.; orbits * orbits];
        for f in 0..orbits {
            for c in 0..orbits {
                forward[f * orbits + c] = weight(c) * cosine(f, c);
                inverse[c * orbits + f] = weight(f) * cosine(f, c) / q as f64;
            }
        }
        Ok(Self {
            orbits,
            forward,
            inverse,
        })
    }
    /// `out[cell * orbits + a] = sum_b matrix[a][b] * input[cell * orbits + b]`.
    fn rows(&self, matrix: &[f64], input: &[f64]) -> Vec<f64> {
        let o = self.orbits;
        let mut out = vec![0.; input.len()];
        for (source, target) in input.chunks_exact(o).zip(out.chunks_exact_mut(o)) {
            for (value, row) in target.iter_mut().zip(matrix.chunks_exact(o)) {
                *value = row.iter().zip(source).map(|(m, x)| m * x).sum();
            }
        }
        out
    }
    /// Frequency-major moments `f * cells + cell`, f = 0..orbits-1, of a
    /// cell-major latent half. Each frequency is a column-major side x side matrix.
    pub fn moments(&self, half: &[f64], cells: usize) -> Result<Vec<f64>> {
        if half.len() != cells * self.orbits || half.iter().any(|x| !x.is_finite()) {
            return Err("invalid latent half for the cosine transform".into());
        }
        let by_cell = self.rows(&self.forward, half);
        let mut out = vec![0.; by_cell.len()];
        for (cell, row) in by_cell.chunks_exact(self.orbits).enumerate() {
            for (f, &value) in row.iter().enumerate() {
                out[f * cells + cell] = value;
            }
        }
        Ok(out)
    }
    /// The cell-major latent half with the given frequency-major moments.
    pub fn latent(&self, moments: &[f64], cells: usize) -> Result<Vec<f64>> {
        if moments.len() != cells * self.orbits || moments.iter().any(|x| !x.is_finite()) {
            return Err("invalid moments for the inverse cosine transform".into());
        }
        let mut by_cell = vec![0.; moments.len()];
        for (f, matrix) in moments.chunks_exact(cells).enumerate() {
            for (cell, &value) in matrix.iter().enumerate() {
                by_cell[cell * self.orbits + f] = value;
            }
        }
        Ok(self.rows(&self.inverse, &by_cell))
    }
}

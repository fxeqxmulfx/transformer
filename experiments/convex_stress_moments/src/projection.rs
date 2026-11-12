//! Euclidean simplex and complex moment-ball projections. The full DFT is a
//! scaled isometry, so independent singular-value clipping projects exactly onto
//! the moment constraints in exact arithmetic. Floating-point defects are tested.
use crate::{C, Model, Result, fft::Fourier, linalg};
use rayon::prelude::*;

pub fn simplex(input: &[f64], output: &mut [f64]) {
    assert_eq!(input.len(), output.len());
    assert!(!input.is_empty() && input.iter().all(|x| x.is_finite()));
    let mut sorted = input.to_vec();
    sorted.sort_unstable_by(|a, b| b.total_cmp(a));
    let (mut sum, mut threshold) = (0., 0.);
    for (j, &x) in sorted.iter().enumerate() {
        sum += x;
        let candidate = (sum - 1.) / (j + 1) as f64;
        if x > candidate {
            threshold = candidate;
        }
    }
    for (out, &x) in output.iter_mut().zip(input) {
        *out = (x - threshold).max(0.);
    }
}

pub struct Projector {
    pub n: usize,
    fft: Fourier,
    pool: rayon::ThreadPool,
}
impl Projector {
    pub fn new(n: usize, workers: usize) -> Result<Self> {
        if workers == 0 {
            return Err("positive worker count required".into());
        }
        Ok(Self {
            n,
            fft: Fourier::new(n)?,
            pool: rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()?,
        })
    }
    pub fn ball(&mut self, p: &[f64], radius: f64) -> Result<Vec<f64>> {
        if !radius.is_finite() || radius <= 0. {
            return Err("invalid radius".into());
        }
        let mut moments = self.fft.forward(p)?;
        let (n, cells) = (self.n, self.n * self.n);
        self.pool.install(|| -> Result<()> {
            moments[cells..(n / 2 + 1) * cells]
                .par_chunks_mut(cells)
                .try_for_each(|m| -> Result<()> {
                    // Frobenius norm upper-bounds the operator norm: this skip is safe.
                    if m.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt() > radius {
                        let svd = linalg::svd(m, n)?;
                        if svd.values[0] > radius {
                            m.copy_from_slice(&svd.reconstruct(radius));
                        }
                    }
                    Ok(())
                })
        })?;
        for f in 1..=n / 2 {
            for cell in 0..cells {
                if 2 * f == n {
                    moments[f * cells + cell].im = 0.;
                } else {
                    moments[(n - f) * cells + cell] = moments[f * cells + cell].conj();
                }
            }
        }
        self.fft.inverse(&moments)
    }
    pub fn norms(&mut self, p: &[f64]) -> Result<Vec<f64>> {
        let moments = self.fft.forward(p)?;
        let (n, cells) = (self.n, self.n * self.n);
        let half: Vec<f64> = self.pool.install(|| {
            moments[cells..(n / 2 + 1) * cells]
                .par_chunks(cells)
                .map(|m| Ok(linalg::svd(m, n)?.values[0]))
                .collect::<Result<Vec<_>>>()
        })?;
        Ok((1..n).map(|f| half[f.min(n - f) - 1]).collect())
    }
    /// CM4: one positive uniform mixture gives a feasible reporting/saved model.
    /// The returned alpha and pre-repair norms expose the amount of correction.
    pub fn feasible(&mut self, p: &[f64], radius: f64) -> Result<(Model, f64, Vec<f64>)> {
        let norms = self.norms(p)?;
        let norm = norms.iter().copied().fold(0., f64::max);
        let alpha = if norm > radius * (1. - 2e-12) {
            radius * (1. - 2e-12) / norm
        } else {
            1.
        };
        let mut model = Model::new(self.n, radius)?;
        for (out, &x) in model.probabilities.iter_mut().zip(p) {
            *out = alpha * x + (1. - alpha) / self.n as f64;
        }
        model.validate_shape()?;
        Ok((model, alpha, norms))
    }
}

pub fn distance(x: &[f64], y: &[f64]) -> f64 {
    x.iter()
        .zip(y)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}

pub fn direct_moments(p: &[f64], n: usize) -> Vec<C> {
    let mut out = vec![C::default(); p.len()];
    for f in 0..n {
        for cell in 0..n * n {
            out[f * n * n + cell] = (0..n)
                .map(|c| {
                    C::from_polar(
                        p[cell * n + c],
                        std::f64::consts::TAU * (f * c) as f64 / n as f64,
                    )
                })
                .sum();
        }
    }
    out
}

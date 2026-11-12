//! CM5: project each complex moment's singular values onto a nonnegative box
//! with a sum budget. The DFT is a scaled isometry; conjugate modes are paired.
//! LAPACK/FFTW and the simplex implementation are inherited from 2de90819.
use crate::{Model, Result, fft::Fourier, linalg};
pub use convex_stress_moments::projection::{distance, simplex};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Norms {
    pub operator: f64,
    pub nuclear: f64,
}

/// Euclidean projection of nonnegative singular values onto 0<=s<=R, sum(s)<=B.
/// If the box projection exceeds B, solve sum(min(R,max(s-tau,0)))=B. The upper
/// bisection endpoint has sum at most B; final mixture validation covers roundoff.
pub fn singular_budget(values: &[f64], radius: f64, budget: f64) -> Result<Vec<f64>> {
    if values.is_empty()
        || values.iter().any(|s| !s.is_finite() || *s < 0.)
        || !radius.is_finite()
        || radius <= 0.
        || !budget.is_finite()
        || budget <= 0.
    {
        return Err("invalid singular-value budget".into());
    }
    let clipped: Vec<_> = values.iter().map(|s| s.min(radius)).collect();
    if clipped.iter().sum::<f64>() <= budget {
        return Ok(clipped);
    }
    let (mut lo, mut hi) = (0., values.iter().copied().fold(0., f64::max));
    for _ in 0..64 {
        let mid = (lo + hi) * 0.5;
        let sum: f64 = values.iter().map(|s| (s - mid).max(0.).min(radius)).sum();
        if sum > budget {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(values
        .iter()
        .map(|s| (s - hi).max(0.).min(radius))
        .collect())
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
    pub fn ball(&mut self, p: &[f64], radius: f64, budget: f64) -> Result<Vec<f64>> {
        if !radius.is_finite() || radius <= 0. || !budget.is_finite() || budget <= 0. {
            return Err("invalid spectral budgets".into());
        }
        let mut moments = self.fft.forward(p)?;
        let (n, cells) = (self.n, self.n * self.n);
        self.pool.install(|| -> Result<()> {
            moments[cells..(n / 2 + 1) * cells]
                .par_chunks_mut(cells)
                .try_for_each(|m| -> Result<()> {
                    let frobenius = m.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
                    // ||M||op<=||M||F and ||M||nuc<=sqrt(n)||M||F.
                    if frobenius > radius || (n as f64).sqrt() * frobenius > budget {
                        let mut svd = linalg::svd(m, n)?;
                        let projected = singular_budget(&svd.values, radius, budget)?;
                        if svd.values != projected {
                            // The inherited public singular-value vector is replaced,
                            // retaining its matching U and V*. Infinite clipping applies
                            // precisely the newly projected values during reconstruction.
                            svd.values = projected;
                            m.copy_from_slice(&svd.reconstruct(f64::INFINITY));
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
    pub fn norms(&mut self, p: &[f64]) -> Result<Vec<Norms>> {
        let moments = self.fft.forward(p)?;
        let (n, cells) = (self.n, self.n * self.n);
        let half = self.pool.install(|| {
            moments[cells..(n / 2 + 1) * cells]
                .par_chunks(cells)
                .map(|m| -> Result<Norms> {
                    let values = linalg::svd(m, n)?.values;
                    Ok(Norms {
                        operator: values[0],
                        nuclear: values.iter().sum(),
                    })
                })
                .collect::<Result<Vec<_>>>()
        })?;
        Ok((1..n).map(|f| half[f.min(n - f) - 1]).collect())
    }
    pub fn feasible(
        &mut self,
        p: &[f64],
        radius: f64,
        budget: f64,
    ) -> Result<(Model, f64, Vec<Norms>)> {
        let norms = self.norms(p)?;
        let ratio = norms
            .iter()
            .map(|s| (s.operator / radius).max(s.nuclear / budget))
            .fold(0., f64::max);
        let alpha = if ratio > 1. - 2e-12 {
            (1. - 2e-12) / ratio
        } else {
            1.
        };
        let mut model = Model::new(self.n, radius, budget)?;
        for (out, &x) in model.probabilities.iter_mut().zip(p) {
            *out = alpha * x + (1. - alpha) / self.n as f64;
        }
        model.validate_shape()?;
        Ok((model, alpha, norms))
    }
}

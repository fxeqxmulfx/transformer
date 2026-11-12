//! Full-state variant of convex_stress_schatten::projection at 7d299b04.
//! The input side determines matrix dimensions; latent q determines DFT length.
//! Conjugate modes share the same projection, including the real even-q mode.
use crate::*;
pub use convex_stress_schatten::projection::{Norms, simplex, singular_budget};

pub struct Projector {
    shape: Shape,
    fft: fft::Fourier,
}
impl Projector {
    pub fn new(shape: Shape) -> Result<Self> {
        memory_bound(shape, 0, 0)?;
        Ok(Self {
            shape,
            fft: fft::Fourier::new(shape.side, shape.states)?,
        })
    }
    pub fn ball(&mut self, point: &[f64]) -> Result<(Vec<f64>, u64)> {
        let (n, q, cells) = (self.shape.side, self.shape.states, self.shape.cells());
        let mut moments = self.fft.forward(point)?;
        let mut decompositions = 0;
        for f in 1..=q / 2 {
            let m = &mut moments[f * cells..(f + 1) * cells];
            let frobenius = m.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
            if frobenius > n as f64 || (n as f64).sqrt() * frobenius > n as f64 {
                let mut svd = linalg::svd(m, n)?;
                decompositions += 1;
                let values = singular_budget(&svd.values, n as f64, n as f64)?;
                if values != svd.values {
                    svd.values = values;
                    m.copy_from_slice(&svd.reconstruct(f64::INFINITY));
                }
            }
        }
        for f in 1..=q / 2 {
            for cell in 0..cells {
                if 2 * f == q {
                    moments[f * cells + cell].im = 0.;
                } else {
                    moments[(q - f) * cells + cell] = moments[f * cells + cell].conj();
                }
            }
        }
        Ok((self.fft.inverse(&moments)?, decompositions))
    }
    /// Check every nonconstant matrix separately, including conjugate modes.
    pub fn norms(&mut self, point: &[f64]) -> Result<Vec<Norms>> {
        let moments = self.fft.forward(point)?;
        moments[self.shape.cells()..]
            .chunks_exact(self.shape.cells())
            .map(|m| {
                let values = linalg::svd(m, self.shape.side)?.values;
                Ok(Norms {
                    operator: values[0],
                    nuclear: values.iter().sum(),
                })
            })
            .collect()
    }
}

pub fn meets(shape: Shape, norms: &[Norms]) -> bool {
    norms.len() == shape.states - 1
        && norms.iter().all(|x| {
            x.operator.is_finite()
                && x.nuclear.is_finite()
                && x.operator >= 0.
                && x.nuclear >= x.operator
                && x.operator <= shape.side as f64 * (1. + 1e-10)
                && x.nuclear <= shape.side as f64 * (1. + 1e-10)
        })
}

/// Project onto the full simplex, or the sub-simplex of a certified class.
pub(crate) fn row(input: &[f64], fiber: Option<&[usize]>, output: &mut [f64]) {
    if let Some(fiber) = fiber {
        let source: Vec<_> = fiber.iter().map(|&z| input[z]).collect();
        let mut target = vec![0.; source.len()];
        simplex(&source, &mut target);
        output.fill(0.);
        for (&z, &x) in fiber.iter().zip(&target) {
            output[z] = x;
        }
    } else {
        simplex(input, output);
    }
}

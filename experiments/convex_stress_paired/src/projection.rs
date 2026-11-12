//! The two convex sets of CM16 in half coordinates, with their projections in
//! the Euclidean metric of the full latent table (an orbit {c, -c} counts twice).
//! A: every row is an even probability vector, optionally supported on given
//! orbits. B: every nonconstant cosine moment meets the operator and nuclear
//! budgets. Both sets are invariant under the reflection, which is an isometry,
//! so the projection of an even table is even and is what is computed here.
//! The transform is a scaled isometry and B constrains each frequency alone, so
//! projecting the singular values of each moment projects onto B.
use crate::{C, Decoder, Model, Result, Shape, linalg, transform::Cosine, weight};
pub use convex_stress_relabel::projection::{Norms, simplex, singular_budget};

/// Latent-metric distance of two even tables given by their halves.
pub fn distance(shape: Shape, x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    x.chunks_exact(shape.orbits)
        .zip(y.chunks_exact(shape.orbits))
        .flat_map(|(a, b)| {
            a.iter()
                .zip(b)
                .enumerate()
                .map(|(c, (p, q))| weight(c) * (p - q) * (p - q))
        })
        .sum::<f64>()
        .sqrt()
}

/// Project an even row onto the even probability vectors supported on the
/// orbits `support` (all orbits if `None`): the simplex projection of the row
/// written out state by state, read back by orbit.
pub fn simplex_even(half: &[f64], support: Option<&[usize]>, out: &mut [f64]) {
    assert_eq!(half.len(), out.len());
    let all: Vec<usize>;
    let orbits = match support {
        Some(orbits) => orbits,
        None => {
            all = (0..half.len()).collect();
            &all
        }
    };
    let mut states: Vec<f64> = orbits.iter().map(|&c| half[c]).collect();
    states.extend(orbits.iter().filter(|&&c| c > 0).map(|&c| half[c]));
    let mut projected = vec![0.; states.len()];
    simplex(&states, &mut projected);
    out.fill(0.);
    for (&c, &p) in orbits.iter().zip(&projected) {
        out[c] = p;
    }
}

pub struct Projector {
    pub shape: Shape,
    cosine: Cosine,
}
impl Projector {
    pub fn new(shape: Shape) -> Result<Self> {
        shape.validate()?;
        Ok(Self {
            shape,
            cosine: Cosine::new(shape.orbits)?,
        })
    }
    fn complex(matrix: &[f64]) -> Vec<C> {
        matrix.iter().map(|&x| C::new(x, 0.)).collect()
    }
    /// Project onto B. Also returns the number of singular value decompositions.
    pub fn ball(&self, half: &[f64], radius: f64, budget: f64) -> Result<(Vec<f64>, u64)> {
        if !radius.is_finite() || radius <= 0. || !budget.is_finite() || budget <= 0. {
            return Err("invalid spectral budgets".into());
        }
        let (side, cells) = (self.shape.side, self.shape.cells());
        let mut moments = self.cosine.moments(half, cells)?;
        let mut decompositions = 0;
        for m in moments[cells..].chunks_exact_mut(cells) {
            let frobenius = m.iter().map(|x| x * x).sum::<f64>().sqrt();
            // ||M||op <= ||M||F and ||M||nuc <= sqrt(side) ||M||F: this skip is safe.
            if frobenius > radius || (side as f64).sqrt() * frobenius > budget {
                let mut svd = linalg::svd(&Self::complex(m), side)?;
                decompositions += 1;
                let projected = singular_budget(&svd.values, radius, budget)?;
                if svd.values != projected {
                    // The projection of a real matrix is real; the imaginary
                    // parts dropped here are roundoff of the complex routine.
                    svd.values = projected;
                    for (out, z) in m.iter_mut().zip(svd.reconstruct(f64::INFINITY)) {
                        *out = z.re;
                    }
                }
            }
        }
        Ok((self.cosine.latent(&moments, cells)?, decompositions))
    }
    /// Operator and nuclear norms of the moments f = 1..orbits-1. The moments
    /// of the frequencies f and -f of an even table coincide.
    pub fn norms(&self, half: &[f64]) -> Result<Vec<Norms>> {
        let (side, cells) = (self.shape.side, self.shape.cells());
        let moments = self.cosine.moments(half, cells)?;
        moments[cells..]
            .chunks_exact(cells)
            .map(|m| {
                let values = linalg::svd(&Self::complex(m), side)?.values;
                Ok(Norms {
                    operator: values[0],
                    nuclear: values.iter().sum(),
                })
            })
            .collect()
    }
    /// Mix with the uniform latent table, whose nonconstant moments vanish,
    /// just enough to meet both budgets; rows stay even probability vectors.
    pub fn feasible(
        &self,
        decoder: &Decoder,
        half: &[f64],
        radius: f64,
        budget: f64,
    ) -> Result<(Model, f64, Vec<Norms>)> {
        let norms = self.norms(half)?;
        let ratio = norms
            .iter()
            .map(|s| (s.operator / radius).max(s.nuclear / budget))
            .fold(0., f64::max);
        let alpha = if ratio > 1. - 2e-12 {
            (1. - 2e-12) / ratio
        } else {
            1.
        };
        let uniform = 1. / self.shape.latent_states() as f64;
        let latent = half
            .iter()
            .map(|&x| alpha * x + (1. - alpha) * uniform)
            .collect();
        let model = Model::new(self.shape, decoder.clone(), radius, budget, latent)?;
        Ok((model, alpha, norms))
    }
}

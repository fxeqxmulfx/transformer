//! Native update primitives for the complete learned complex sequence family.
//!
//! Source: `review/flat_spectrum.md`, CM75--CM81, especially the checked Lean
//! `LearnedHinge.Projection` certificates. No training or accuracy is asserted
//! by this library component. Exact rational arithmetic verifies feasibility;
//! bounded active-set search refuses rather than weakening its constraints.

pub mod domain;
pub mod geometry;
pub mod model;
pub mod projection;

use num_rational::BigRational;

pub type Rational = BigRational;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refusal {
    Shape,
    EmptyObjects,
    InvalidObject,
    ContextTooLong,
    InvalidPattern,
    Overflow,
    Budget,
    NonFinite,
    NegativeMultiplier,
    Infeasible,
    Complementarity,
    Stationarity,
    NoCertificate,
    InvalidRenaming,
    NumericalUnderflow,
    GeometryCertificate,
    BasisKernel,
    BasisCoordinates,
    BasisRecovery,
    QueryDirection,
    QueryStep,
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for Refusal {}

/// Explicit search and rational-size budgets; no implicit pattern sampling.
#[derive(Clone, Copy, Debug)]
pub struct ProjectionLimits {
    pub max_active_sets: usize,
    pub max_matrix_entries: usize,
    pub max_coefficient_bits: u64,
}

impl ProjectionLimits {
    pub fn check(&self, value: &Rational) -> Result<(), Refusal> {
        if value.numer().bits() > self.max_coefficient_bits
            || value.denom().bits() > self.max_coefficient_bits
        {
            return Err(Refusal::Budget);
        }
        Ok(())
    }

    /// Preserve the exact dyadic value of a finite binary64 proposal.
    pub fn from_f64(&self, value: f64) -> Result<Rational, Refusal> {
        let exact = Rational::from_float(value).ok_or(Refusal::NonFinite)?;
        self.check(&exact)?;
        Ok(exact)
    }
}

pub(crate) fn checked_dot(
    a: &[Rational],
    b: &[Rational],
    limits: ProjectionLimits,
) -> Result<Rational, Refusal> {
    use num_traits::Zero;
    if a.len() != b.len() {
        return Err(Refusal::Shape);
    }
    let mut total = Rational::zero();
    for (left, right) in a.iter().zip(b) {
        let product = left * right;
        limits.check(&product)?;
        total += product;
        limits.check(&total)?;
    }
    Ok(total)
}

#[cfg(test)]
mod tests;

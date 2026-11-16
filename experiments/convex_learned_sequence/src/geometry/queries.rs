//! Exact rational CM80 query bounds and mixtures, retaining all inequalities.
//! The resulting points are inputs for the actual complex forward. No finite
//! difference accuracy, gradient identity or descent claim is encoded here.

use super::ConeGeometry;
use crate::{Rational, Refusal, checked_dot};
use num_traits::{One, Signed, Zero};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SymmetricQuery {
    pub step: Rational,
    pub plus: Vec<Rational>,
    pub minus: Vec<Rational>,
}

impl ConeGeometry {
    pub fn feasible(&self, point: &[Rational]) -> Result<(), Refusal> {
        if point.len() != self.problem.dimension() {
            return Err(Refusal::Shape);
        }
        let limits = self.problem.limits();
        for value in point {
            limits.check(value)?;
        }
        for normal in self.problem.normals() {
            if checked_dot(normal, point, limits)? < Rational::zero() {
                return Err(Refusal::Infeasible);
            }
        }
        Ok(())
    }

    pub fn combine(&self, coefficients: &[Rational]) -> Result<Vec<Rational>, Refusal> {
        if coefficients.len() != self.basis().vectors.len() {
            return Err(Refusal::Shape);
        }
        let limits = self.problem.limits();
        let mut result = vec![Rational::zero(); self.problem.dimension()];
        for (coefficient, vector) in coefficients.iter().zip(&self.basis().vectors) {
            limits.check(coefficient)?;
            for (value, entry) in result.iter_mut().zip(vector) {
                let term = coefficient * entry;
                limits.check(&term)?;
                *value += term;
                limits.check(value)?;
            }
        }
        self.verify_direction(&result)?;
        Ok(result)
    }

    /// Kernel membership and reconstruction both use the checked full basis.
    pub fn verify_direction(&self, direction: &[Rational]) -> Result<(), Refusal> {
        if direction.len() != self.problem.dimension() {
            return Err(Refusal::Shape);
        }
        let limits = self.problem.limits();
        for value in direction {
            limits.check(value)?;
        }
        for row in &self.equalities {
            if !checked_dot(row, direction, limits)?.is_zero() {
                return Err(Refusal::QueryDirection);
            }
        }
        for (d, value) in direction.iter().enumerate() {
            let mut reconstructed = Rational::zero();
            for (basis, &free) in self
                .basis()
                .vectors
                .iter()
                .zip(&self.basis().free_coordinates)
            {
                let term = &basis[d] * &direction[free];
                limits.check(&term)?;
                reconstructed += term;
                limits.check(&reconstructed)?;
            }
            if value != &reconstructed {
                return Err(Refusal::BasisRecovery);
            }
        }
        Ok(())
    }

    pub fn mixture(&self, point: &[Rational], amount: &Rational) -> Result<Vec<Rational>, Refusal> {
        self.feasible(point)?;
        let limits = self.problem.limits();
        limits.check(amount)?;
        if amount <= &Rational::zero() || amount > &Rational::one() {
            return Err(Refusal::QueryStep);
        }
        let complement = Rational::one() - amount;
        limits.check(&complement)?;
        let mut result = Vec::with_capacity(point.len());
        for (old, centre) in point.iter().zip(self.centre()) {
            let left = &complement * old;
            let right = amount * centre;
            limits.check(&left)?;
            limits.check(&right)?;
            let value = left + right;
            limits.check(&value)?;
            result.push(value);
        }
        self.feasible(&result)?;
        Ok(result)
    }

    /// The cap is explicit and positive. Every original row contributes its
    /// CM80 bound; a zero slack with nonzero response is an explicit refusal.
    pub fn symmetric_queries(
        &self,
        point: &[Rational],
        direction: &[Rational],
        cap: &Rational,
    ) -> Result<SymmetricQuery, Refusal> {
        self.feasible(point)?;
        self.verify_direction(direction)?;
        let limits = self.problem.limits();
        limits.check(cap)?;
        if cap <= &Rational::zero() {
            return Err(Refusal::QueryStep);
        }
        let mut step = cap.clone();
        for row in self.problem.normals() {
            let slack = checked_dot(row, point, limits)?;
            let slope = checked_dot(row, direction, limits)?;
            if slack.is_zero() {
                if !slope.is_zero() {
                    return Err(Refusal::QueryDirection);
                }
            } else {
                let denominator = slope.abs() + Rational::one();
                limits.check(&denominator)?;
                let bound = slack / denominator;
                limits.check(&bound)?;
                step = step.min(bound);
            }
        }
        if step <= Rational::zero() {
            return Err(Refusal::QueryStep);
        }
        let mut plus = Vec::with_capacity(point.len());
        let mut minus = Vec::with_capacity(point.len());
        for (value, slope) in point.iter().zip(direction) {
            let delta = &step * slope;
            limits.check(&delta)?;
            let upper = value + &delta;
            let lower = value - &delta;
            limits.check(&upper)?;
            limits.check(&lower)?;
            plus.push(upper);
            minus.push(lower);
        }
        self.feasible(&plus)?;
        self.feasible(&minus)?;
        // Independently replay all original absolute-response bounds.
        for row in self.problem.normals() {
            let change = &step * checked_dot(row, direction, limits)?.abs();
            limits.check(&change)?;
            if change > checked_dot(row, point, limits)? {
                return Err(Refusal::QueryStep);
            }
        }
        Ok(SymmetricQuery { step, plus, minus })
    }
}

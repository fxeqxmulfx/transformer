//! Recover a complete kernel vector from every supplied basis pairing.
//! CM81 `pairing_agreement` (Lean 2d2ca3ae) justifies extension from exact
//! basis pairing equality to all kernel directions. Supplied numerical
//! finite differences are not asserted to be exact loss derivatives here.

use super::ConeGeometry;
use crate::{Rational, Refusal, checked_dot, projection::solve};

impl ConeGeometry {
    /// Use the full Gram system, including its off-diagonal entries. A
    /// nonorthogonal basis's pairings are not its expansion coefficients.
    pub fn recover_pairings(&self, pairings: &[Rational]) -> Result<Vec<Rational>, Refusal> {
        let basis = &self.basis().vectors;
        if pairings.len() != basis.len() {
            return Err(Refusal::Shape);
        }
        let limits = self.problem.limits();
        for value in pairings {
            limits.check(value)?;
        }
        // GeometryPlan already reserves complete quadratic basis/Gram work.
        // Every original coordinate participates in every Gram inner product.
        let mut matrix = Vec::with_capacity(basis.len());
        for (row, pairing) in basis.iter().zip(pairings) {
            let mut values = Vec::with_capacity(basis.len() + 1);
            for column in basis {
                values.push(checked_dot(row, column, limits)?);
            }
            values.push(pairing.clone());
            matrix.push(values);
        }
        let coefficients = solve(matrix, limits)?.ok_or(Refusal::NoCertificate)?;
        let vector = self.combine(&coefficients)?;
        self.verify_pairings(&vector, pairings)?;
        Ok(vector)
    }

    /// Independent acceptance checks the complete kernel reconstruction and
    /// every supplied pairing. It does not call the Gram-system generator.
    pub fn verify_pairings(
        &self,
        vector: &[Rational],
        pairings: &[Rational],
    ) -> Result<(), Refusal> {
        if pairings.len() != self.basis().vectors.len() {
            return Err(Refusal::Shape);
        }
        self.verify_direction(vector)?;
        let limits = self.problem.limits();
        for (basis, expected) in self.basis().vectors.iter().zip(pairings) {
            limits.check(expected)?;
            if &checked_dot(basis, vector, limits)? != expected {
                return Err(Refusal::QueryPairing);
            }
        }
        Ok(())
    }
}

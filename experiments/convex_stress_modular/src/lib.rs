//! CM27 / Lean98b12abc: exact residual lifting and Bezout recombination.
//! Integer checks certify supplied congruences, not categorical discovery or
//! true unseen labels. Native class-code discovery is a separate next layer.
pub use convex_stress_fibers::{Model, Result, Sample, finite_system};
pub mod lift;
pub mod recombine;
use serde::{Deserialize, Serialize};
#[cfg(test)]
mod tests;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct System {
    pub variables: usize,
    pub rows: Vec<Vec<i64>>,
    pub right: Vec<i64>,
}
impl System {
    pub fn validate(&self) -> Result<()> {
        if !(1..=1024).contains(&self.variables)
            || self.rows.is_empty()
            || self.rows.len() > 8192
            || self.right.len() != self.rows.len()
            || self.right.iter().any(|x| x.unsigned_abs() > 1_073_741_824)
            || self
                .rows
                .iter()
                .any(|r| r.len() != self.variables || r.iter().any(|x| x.unsigned_abs() > 1024))
        {
            return Err(
                "integer constraint system exceeds bounded shape or coefficient ranges".into(),
            );
        }
        Ok(())
    }
    pub fn validate_point(&self, modulus: u16, point: &[u16]) -> Result<()> {
        self.validate()?;
        if !(1..=257).contains(&modulus)
            || point.len() != self.variables
            || point.iter().any(|&x| x >= modulus)
        {
            return Err("invalid canonical integer representatives or modulus".into());
        }
        Ok(())
    }
    pub(crate) fn residuals(&self, point: &[u16]) -> Vec<i64> {
        self.rows
            .iter()
            .zip(&self.right)
            .map(|(r, &b)| {
                b - r
                    .iter()
                    .zip(point)
                    .map(|(&a, &x)| a * x as i64)
                    .sum::<i64>()
            })
            .collect()
    }
}

/// Independent original-row arithmetic uses i128 and checks each coefficient
/// product directly; it does not call the producer's i64 residual routine.
pub(crate) fn checked_residuals(s: &System, modulus: u16, x: &[u16]) -> Result<Vec<i64>> {
    s.validate_point(modulus, x)?;
    let mut values = vec![];
    for (row, &b) in s.rows.iter().zip(&s.right) {
        let mut value = b as i128;
        for (&a, &z) in row.iter().zip(x) {
            value -= a as i128 * z as i128;
        }
        if value % (modulus as i128) != 0 {
            return Err("original integer equation fails its claimed modulus".into());
        }
        values.push(i64::try_from(value)?);
    }
    Ok(values)
}

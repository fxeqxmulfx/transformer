//! CM81 finite basis certificates (Lean commit 2d2ca3ae).
//! Generation retains the full equality matrix and its full row transform.
//! Verification checks the original identities independently of elimination.

use crate::{ProjectionLimits, Rational, Refusal, checked_dot};
use num_traits::{One, Zero};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BasisCertificate {
    pub vectors: Vec<Vec<Rational>>,
    pub free_coordinates: Vec<usize>,
    pub recovery: Vec<Vec<Rational>>,
}

pub(super) fn generate(
    dimension: usize,
    rows: &[Vec<Rational>],
    limits: ProjectionLimits,
) -> Result<BasisCertificate, Refusal> {
    let count = rows.len();
    let mut reduced = rows.to_vec();
    let mut transform = vec![vec![Rational::zero(); count]; count];
    for (i, row) in transform.iter_mut().enumerate() {
        row[i] = Rational::one();
    }
    let mut pivots = Vec::new();
    for column in 0..dimension {
        let rank = pivots.len();
        let Some(pivot) = (rank..count).find(|&r| !reduced[r][column].is_zero()) else {
            continue;
        };
        reduced.swap(rank, pivot);
        transform.swap(rank, pivot);
        let scale = reduced[rank][column].clone();
        for value in reduced[rank].iter_mut().chain(&mut transform[rank]) {
            *value /= &scale;
            limits.check(value)?;
        }
        let reduced_pivot = reduced[rank].clone();
        let transform_pivot = transform[rank].clone();
        for row in 0..count {
            if row == rank {
                continue;
            }
            let factor = reduced[row][column].clone();
            for (value, base) in reduced[row]
                .iter_mut()
                .chain(&mut transform[row])
                .zip(reduced_pivot.iter().chain(&transform_pivot))
            {
                let change = &factor * base;
                limits.check(&change)?;
                *value -= change;
                limits.check(value)?;
            }
        }
        pivots.push(column);
    }
    let free_coordinates: Vec<_> = (0..dimension).filter(|d| !pivots.contains(d)).collect();
    let mut vectors = vec![vec![Rational::zero(); dimension]; free_coordinates.len()];
    for (vector, &free) in vectors.iter_mut().zip(&free_coordinates) {
        vector[free] = Rational::one();
        for (row, &pivot) in pivots.iter().enumerate() {
            vector[pivot] = -&reduced[row][free];
        }
    }
    let mut recovery = vec![vec![Rational::zero(); count]; dimension];
    for (row, &pivot) in pivots.iter().enumerate() {
        recovery[pivot] = transform[row].clone();
    }
    let result = BasisCertificate {
        vectors,
        free_coordinates,
        recovery,
    };
    verify(dimension, rows, &result, limits)?;
    Ok(result)
}

pub(super) fn verify(
    dimension: usize,
    rows: &[Vec<Rational>],
    result: &BasisCertificate,
    limits: ProjectionLimits,
) -> Result<(), Refusal> {
    if rows.iter().any(|row| row.len() != dimension)
        || result.vectors.len() != result.free_coordinates.len()
        || result.vectors.len() > dimension
        || result.vectors.iter().any(|row| row.len() != dimension)
        || result.recovery.len() != dimension
        || result.recovery.iter().any(|row| row.len() != rows.len())
        || result.free_coordinates.iter().any(|&d| d >= dimension)
    {
        return Err(Refusal::Shape);
    }
    for (j, &free) in result.free_coordinates.iter().enumerate() {
        if result.free_coordinates[..j].contains(&free) {
            return Err(Refusal::BasisCoordinates);
        }
    }
    for value in rows
        .iter()
        .chain(&result.vectors)
        .chain(&result.recovery)
        .flatten()
    {
        limits.check(value)?;
    }
    for (j, vector) in result.vectors.iter().enumerate() {
        for row in rows {
            if !checked_dot(row, vector, limits)?.is_zero() {
                return Err(Refusal::BasisKernel);
            }
        }
        for (k, &free) in result.free_coordinates.iter().enumerate() {
            let expected = if j == k {
                Rational::one()
            } else {
                Rational::zero()
            };
            if vector[free] != expected {
                return Err(Refusal::BasisCoordinates);
            }
        }
    }
    // Every original coordinate identity is checked against every original
    // equality row. There is no rank claim trusted from the generator.
    for (d, coefficients) in result.recovery.iter().enumerate() {
        for coordinate in 0..dimension {
            let mut residual = if d == coordinate {
                Rational::one()
            } else {
                Rational::zero()
            };
            for (vector, &free) in result.vectors.iter().zip(&result.free_coordinates) {
                if coordinate == free {
                    residual -= &vector[d];
                    limits.check(&residual)?;
                }
            }
            let mut recovered = Rational::zero();
            for (coefficient, row) in coefficients.iter().zip(rows) {
                let term = coefficient * &row[coordinate];
                limits.check(&term)?;
                recovered += term;
                limits.check(&recovered)?;
            }
            if residual != recovered {
                return Err(Refusal::BasisRecovery);
            }
        }
    }
    Ok(())
}

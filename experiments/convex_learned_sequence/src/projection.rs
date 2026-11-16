//! Exact arithmetic implementation of CM79's finite projection certificate.
//!
//! The active-set search is a finite candidate generator, not a trusted
//! optimizer oracle. Only a candidate passing every original inequality and
//! all four certificate conditions is returned. Singular active systems are
//! not inverted; their original rows remain in every certificate check.

use crate::{ProjectionLimits, Rational, Refusal, checked_dot};
use num_traits::Zero;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Projection {
    pub point: Vec<Rational>,
    pub multipliers: Vec<Rational>,
}

#[derive(Clone, Debug)]
pub struct ProjectionProblem {
    dimension: usize,
    normals: Vec<Vec<Rational>>,
    gram: Vec<Vec<Rational>>,
    active_sets: usize,
    limits: ProjectionLimits,
}

impl ProjectionProblem {
    pub fn new(
        dimension: usize,
        normals: Vec<Vec<Rational>>,
        limits: ProjectionLimits,
    ) -> Result<Self, Refusal> {
        if dimension == 0 || normals.iter().any(|row| row.len() != dimension) {
            return Err(Refusal::Shape);
        }
        let rows = normals.len();
        let active_sets = 1usize
            .checked_shl(u32::try_from(rows).map_err(|_| Refusal::Overflow)?)
            .ok_or(Refusal::Overflow)?;
        // Charge retained rows, the Gram matrix, candidate elimination work,
        // proposal/point/multiplier arrays, and spare rational temporaries.
        let entries = rows
            .checked_mul(dimension)
            .and_then(|n| rows.checked_mul(rows)?.checked_mul(3)?.checked_add(n))
            .and_then(|n| rows.checked_add(dimension)?.checked_mul(8)?.checked_add(n))
            .ok_or(Refusal::Overflow)?;
        if active_sets > limits.max_active_sets
            || entries > limits.max_matrix_entries
            || limits.max_coefficient_bits == 0
        {
            return Err(Refusal::Budget);
        }
        for value in normals.iter().flatten() {
            limits.check(value)?;
        }
        let mut gram = vec![vec![Rational::zero(); rows]; rows];
        for (i, row) in normals.iter().enumerate() {
            for (j, other) in normals.iter().enumerate() {
                gram[i][j] = checked_dot(row, other, limits)?;
            }
        }
        Ok(Self {
            dimension,
            normals,
            gram,
            active_sets,
            limits,
        })
    }

    pub fn normals(&self) -> &[Vec<Rational>] {
        &self.normals
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn limits(&self) -> ProjectionLimits {
        self.limits
    }

    /// Check the exact CM79 predicates, independently of candidate generation.
    pub fn verify(&self, proposal: &[Rational], result: &Projection) -> Result<(), Refusal> {
        if proposal.len() != self.dimension
            || result.point.len() != self.dimension
            || result.multipliers.len() != self.normals.len()
        {
            return Err(Refusal::Shape);
        }
        for value in proposal
            .iter()
            .chain(&result.point)
            .chain(&result.multipliers)
        {
            self.limits.check(value)?;
        }
        for (row, multiplier) in self.normals.iter().zip(&result.multipliers) {
            if multiplier < &Rational::zero() {
                return Err(Refusal::NegativeMultiplier);
            }
            let slack = checked_dot(row, &result.point, self.limits)?;
            if slack < Rational::zero() {
                return Err(Refusal::Infeasible);
            }
            let complement = multiplier * slack;
            self.limits.check(&complement)?;
            if !complement.is_zero() {
                return Err(Refusal::Complementarity);
            }
        }
        for (d, original) in proposal.iter().enumerate() {
            let mut stationarity = original.clone();
            for (row, multiplier) in self.normals.iter().zip(&result.multipliers) {
                let contribution = multiplier * &row[d];
                self.limits.check(&contribution)?;
                stationarity += contribution;
                self.limits.check(&stationarity)?;
            }
            if stationarity != result.point[d] {
                return Err(Refusal::Stationarity);
            }
        }
        Ok(())
    }

    /// Search every active subset as needed; a verified certificate proves
    /// global optimality and permits early return without testing competitors.
    pub fn project(&self, proposal: &[Rational]) -> Result<Projection, Refusal> {
        if proposal.len() != self.dimension {
            return Err(Refusal::Shape);
        }
        for value in proposal {
            self.limits.check(value)?;
        }
        let rhs: Vec<_> = self
            .normals
            .iter()
            .map(|row| checked_dot(row, proposal, self.limits).map(|x| -x))
            .collect::<Result<_, _>>()?;
        for mask in 0..self.active_sets {
            let active: Vec<_> = (0..self.normals.len())
                .filter(|&row| mask & (1usize << row) != 0)
                .collect();
            let matrix: Vec<_> = active
                .iter()
                .map(|&row| {
                    let mut values: Vec<_> = active
                        .iter()
                        .map(|&col| self.gram[row][col].clone())
                        .collect();
                    values.push(rhs[row].clone());
                    values
                })
                .collect();
            let Some(solution) = solve(matrix, self.limits)? else {
                continue;
            };
            let mut result = Projection {
                point: proposal.to_vec(),
                multipliers: vec![Rational::zero(); self.normals.len()],
            };
            for (&row, coefficient) in active.iter().zip(solution) {
                result.multipliers[row] = coefficient;
            }
            for (d, value) in result.point.iter_mut().enumerate() {
                for (row, multiplier) in self.normals.iter().zip(&result.multipliers) {
                    let term = multiplier * &row[d];
                    self.limits.check(&term)?;
                    *value += term;
                    self.limits.check(value)?;
                }
            }
            match self.verify(proposal, &result) {
                Ok(()) => return Ok(result),
                Err(Refusal::NegativeMultiplier | Refusal::Infeasible) => {}
                Err(error) => return Err(error),
            }
        }
        Err(Refusal::NoCertificate)
    }
}

/// Exact Gauss--Jordan elimination of an augmented square system. Shared by
/// projection candidates and basis-pairing recovery; singularity is a refusal
/// of the candidate, never a removed model constraint or query direction.
pub(crate) fn solve(
    mut matrix: Vec<Vec<Rational>>,
    limits: ProjectionLimits,
) -> Result<Option<Vec<Rational>>, Refusal> {
    let size = matrix.len();
    let width = size.checked_add(1).ok_or(Refusal::Overflow)?;
    if matrix.iter().any(|row| row.len() != width) {
        return Err(Refusal::Shape);
    }
    for value in matrix.iter().flatten() {
        limits.check(value)?;
    }
    for column in 0..size {
        let Some(pivot) = (column..size).find(|&row| !matrix[row][column].is_zero()) else {
            return Ok(None);
        };
        matrix.swap(column, pivot);
        let scale = matrix[column][column].clone();
        for entry in &mut matrix[column][column..=size] {
            *entry /= &scale;
            limits.check(entry)?;
        }
        let pivot_row = matrix[column].clone();
        for (row_index, row) in matrix.iter_mut().enumerate() {
            if row_index == column {
                continue;
            }
            let multiplier = row[column].clone();
            for (entry, pivot_entry) in row[column..=size].iter_mut().zip(&pivot_row[column..=size])
            {
                let contribution = &multiplier * pivot_entry;
                limits.check(&contribution)?;
                *entry -= contribution;
                limits.check(entry)?;
            }
        }
    }
    Ok(Some(
        matrix.into_iter().map(|row| row[size].clone()).collect(),
    ))
}

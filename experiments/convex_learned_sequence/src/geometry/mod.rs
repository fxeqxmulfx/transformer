//! Exact CM80--CM81 feasible-query geometry (Lean eb1634bd and 2d2ca3ae).
//! All normal projections and original equality rows are retained. Native
//! generators are independent of finite certificate checks; this library is
//! tested against Lean specifications, not extracted or formally verified Rust.

mod basis;
mod queries;

pub use basis::BasisCertificate;
pub use queries::SymmetricQuery;

use crate::projection::{Projection, ProjectionProblem};
use crate::{Rational, Refusal, checked_dot};
use num_traits::Zero;

#[derive(Clone, Copy, Debug)]
pub struct GeometryLimits {
    pub max_normal_projections: usize,
    pub max_active_candidates: usize,
    pub max_rational_entries: usize,
    pub max_working_bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GeometryPlan {
    pub normal_projections: usize,
    pub max_active_candidates: usize,
    pub rational_entries: usize,
    pub reserved_bytes: usize,
}

impl GeometryPlan {
    pub fn new(problem: &ProjectionProblem, limits: GeometryLimits) -> Result<Self, Refusal> {
        let count = problem.normals().len();
        let candidates = 1usize
            .checked_shl(u32::try_from(count).map_err(|_| Refusal::Overflow)?)
            .and_then(|n| n.checked_mul(count))
            .ok_or(Refusal::Overflow)?;
        let k = count as u128;
        let d = problem.dimension() as u128;
        // Original problem/work, all normal certificates, equality/RREF and
        // row-operation matrices, complete basis/recovery, and replay buffers.
        let entries = 8 * k * d + 8 * k * k + 4 * d * d + 32 * (k + d);
        let bytes_per_entry =
            512 + 8 * u128::from(problem.limits().max_coefficient_bits).div_ceil(8);
        let bytes = entries
            .checked_mul(bytes_per_entry)
            .ok_or(Refusal::Overflow)?;
        let result = Self {
            normal_projections: count,
            max_active_candidates: candidates,
            rational_entries: usize::try_from(entries).map_err(|_| Refusal::Overflow)?,
            reserved_bytes: usize::try_from(bytes).map_err(|_| Refusal::Overflow)?,
        };
        if result.normal_projections > limits.max_normal_projections
            || result.max_active_candidates > limits.max_active_candidates
            || result.rational_entries > limits.max_rational_entries
            || result.reserved_bytes > limits.max_working_bytes
        {
            return Err(Refusal::Budget);
        }
        Ok(result)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GeometryCertificate {
    pub normal_projections: Vec<Projection>,
    pub centre: Vec<Rational>,
    pub forced: Vec<bool>,
    pub basis: BasisCertificate,
}

#[derive(Clone, Debug)]
pub struct ConeGeometry {
    problem: ProjectionProblem,
    certificate: GeometryCertificate,
    equalities: Vec<Vec<Rational>>,
    plan: GeometryPlan,
}

impl ConeGeometry {
    pub fn new(problem: ProjectionProblem, limits: GeometryLimits) -> Result<Self, Refusal> {
        GeometryPlan::new(&problem, limits)?;
        let arithmetic = problem.limits();
        let mut centre = vec![Rational::zero(); problem.dimension()];
        let mut normal_projections = Vec::with_capacity(problem.normals().len());
        for normal in problem.normals() {
            let projection = problem.project(normal)?;
            problem.verify(normal, &projection)?;
            for (value, projected) in centre.iter_mut().zip(&projection.point) {
                *value += projected;
                arithmetic.check(value)?;
            }
            normal_projections.push(projection);
        }
        let mut forced = Vec::with_capacity(problem.normals().len());
        for row in problem.normals() {
            let slack = checked_dot(row, &centre, arithmetic)?;
            if slack < Rational::zero() {
                return Err(Refusal::Infeasible);
            }
            forced.push(slack.is_zero());
        }
        let equalities = equality_rows(&problem, &forced);
        let basis = basis::generate(problem.dimension(), &equalities, arithmetic)?;
        Self::from_certificate(
            problem,
            GeometryCertificate {
                normal_projections,
                centre,
                forced,
                basis,
            },
            limits,
        )
    }

    /// Acceptance replays the finite CM80--CM81 checks independently of both
    /// generators. An externally supplied incomplete/corrupt certificate fails.
    pub fn from_certificate(
        problem: ProjectionProblem,
        certificate: GeometryCertificate,
        limits: GeometryLimits,
    ) -> Result<Self, Refusal> {
        let plan = GeometryPlan::new(&problem, limits)?;
        let equalities = verify(&problem, &certificate)?;
        Ok(Self {
            problem,
            certificate,
            equalities,
            plan,
        })
    }

    pub fn certificate(&self) -> &GeometryCertificate {
        &self.certificate
    }
    pub fn centre(&self) -> &[Rational] {
        &self.certificate.centre
    }
    pub fn basis(&self) -> &BasisCertificate {
        &self.certificate.basis
    }
    pub fn equalities(&self) -> &[Vec<Rational>] {
        &self.equalities
    }
    pub fn normals(&self) -> &[Vec<Rational>] {
        self.problem.normals()
    }
    pub fn plan(&self) -> &GeometryPlan {
        &self.plan
    }

    pub fn verify(&self, certificate: &GeometryCertificate) -> Result<(), Refusal> {
        verify(&self.problem, certificate).map(|_| ())
    }
}

fn equality_rows(problem: &ProjectionProblem, forced: &[bool]) -> Vec<Vec<Rational>> {
    problem
        .normals()
        .iter()
        .zip(forced)
        .map(|(row, &forced)| {
            if forced {
                row.clone()
            } else {
                vec![Rational::zero(); problem.dimension()]
            }
        })
        .collect()
}

fn verify(
    problem: &ProjectionProblem,
    certificate: &GeometryCertificate,
) -> Result<Vec<Vec<Rational>>, Refusal> {
    let count = problem.normals().len();
    let dimension = problem.dimension();
    let arithmetic = problem.limits();
    if certificate.normal_projections.len() != count
        || certificate.centre.len() != dimension
        || certificate.forced.len() != count
    {
        return Err(Refusal::Shape);
    }
    let mut expected_centre = vec![Rational::zero(); dimension];
    for (normal, projection) in problem
        .normals()
        .iter()
        .zip(&certificate.normal_projections)
    {
        problem.verify(normal, projection)?;
        for (value, projected) in expected_centre.iter_mut().zip(&projection.point) {
            *value += projected;
            arithmetic.check(value)?;
        }
    }
    if certificate.centre != expected_centre {
        return Err(Refusal::GeometryCertificate);
    }
    for (r, normal) in problem.normals().iter().enumerate() {
        let slack = checked_dot(normal, &certificate.centre, arithmetic)?;
        if slack < Rational::zero() {
            return Err(Refusal::Infeasible);
        }
        let zero_projection = certificate.normal_projections[r]
            .point
            .iter()
            .all(Rational::is_zero);
        if certificate.forced[r] != slack.is_zero() || zero_projection != slack.is_zero() {
            return Err(Refusal::GeometryCertificate);
        }
    }
    let equalities = equality_rows(problem, &certificate.forced);
    basis::verify(dimension, &equalities, &certificate.basis, arithmetic)?;
    Ok(equalities)
}

#[cfg(test)]
mod tests;

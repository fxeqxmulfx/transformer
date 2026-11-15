//! Training-only homogeneous CM57 constraints. The query-feature span is an
//! observable numerical diagnostic, distinct from unknown-target realizability.
use crate::{data::Record, projection as project, *};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Domain {
    pub radius: f64,
    pub minimum_mass: f64,
    pub budgets: Vec<f64>,
    pub anchors: Vec<Vec<Parameters>>,
    pub basis: Vec<Vec<Parameters>>,
    pub feature_basis: Vec<Vec<Parameters>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Projection {
    pub theta: Vec<Parameters>,
    pub cycles: usize,
    pub normalized_violation: f64,
    pub normalized_step: f64,
    pub squared_distance: f64,
}

impl Domain {
    pub fn new(
        rows: &[Record],
        coordinates: &[f64],
        budgets: Vec<f64>,
        radius: f64,
        minimum_mass: f64,
    ) -> Result<Self> {
        data::records(rows, coordinates, budgets.len())?;
        if budgets
            .iter()
            .any(|b| !b.is_finite() || !(6. ..=1e6).contains(b))
            || !radius.is_finite()
            || !(1e-6..=1e12).contains(&radius)
            || !minimum_mass.is_finite()
            || minimum_mass <= 0.
            || 6. * minimum_mass > radius
        {
            return Err("invalid global, perspective or positive-mass budget".into());
        }
        let mut anchors = vec![vec![[0., 0., 0., 0., 1., -1.]]; budgets.len()];
        for r in rows {
            for (p, h) in r.features.iter().enumerate() {
                anchors[p].push([h[0], h[1], h[2], h[3], h[4], -coordinates[r.label]]);
            }
        }
        let basis = anchors
            .iter()
            .map(|r| project::basis(r))
            .collect::<Result<Vec<_>>>()?;
        let feature_basis = anchors
            .iter()
            .map(|rows| {
                project::basis(
                    &rows
                        .iter()
                        .map(|r| {
                            let mut r = *r;
                            r[5] = 0.;
                            r
                        })
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        if basis.iter().all(|b| b.len() == 6) {
            return Err("all program masses forced to zero by observed constraints".into());
        }
        Ok(Self {
            radius,
            minimum_mass,
            budgets,
            anchors,
            basis,
            feature_basis,
        })
    }

    pub fn validate(&self) -> Result<()> {
        let p = self.budgets.len();
        if !(1..=MAX_PROGRAMS).contains(&p)
            || self.anchors.len() != p
            || self.basis.len() != p
            || self.feature_basis.len() != p
            || !self.radius.is_finite()
            || !(1e-6..=1e12).contains(&self.radius)
            || !self.minimum_mass.is_finite()
            || self.minimum_mass <= 0.
            || 6. * self.minimum_mass > self.radius
            || self
                .budgets
                .iter()
                .any(|b| !b.is_finite() || !(6. ..=1e6).contains(b))
        {
            return Err("invalid saved convex domain".into());
        }
        let count = self.anchors[0].len();
        if !(3..=MAX_RECORDS + 1).contains(&count) {
            return Err("invalid anchor count".into());
        }
        data::forecast(count - 1, p, 2)?;
        for (p, (rows, basis)) in self.anchors.iter().zip(&self.basis).enumerate() {
            if rows.len() != count
                || rows
                    .iter()
                    .flatten()
                    .any(|x| !x.is_finite() || x.abs() > 4096.)
                || rows[0] != [0., 0., 0., 0., 1., -1.]
                || *basis != project::basis(rows)?
            {
                return Err("saved anchors or orthogonal constraint basis changed".into());
            }
            let features: Vec<_> = rows
                .iter()
                .map(|r| {
                    let mut r = *r;
                    r[5] = 0.;
                    r
                })
                .collect();
            if self.feature_basis[p] != project::basis(&features)? {
                return Err("saved observable feature-span basis changed".into());
            }
        }
        Ok(())
    }

    /// Numerical coverage only; zero residual does not establish the unknown
    /// decoder's semantic realizability at the query.
    pub fn span_residual(&self, features: &[Features]) -> Result<Vec<f64>> {
        data::features(features, self.budgets.len())?;
        Ok(features
            .iter()
            .zip(&self.feature_basis)
            .map(|(h, b)| {
                let mut r = [h[0], h[1], h[2], h[3], h[4], 0.];
                project::linear(&mut r, b);
                project::dot(&r, &r).sqrt() / h.iter().map(|x| x * x).sum::<f64>().sqrt().max(1.)
            })
            .collect())
    }

    pub fn violation(&self, theta: &[Parameters]) -> f64 {
        if theta.len() != self.budgets.len() || theta.iter().flatten().any(|x| !x.is_finite()) {
            return f64::INFINITY;
        }
        let mut error = ((charge(theta) - self.radius) / self.radius.max(1.)).max(0.);
        error = error.max(
            (self.minimum_mass - theta.iter().map(|t| t[5]).sum::<f64>())
                / self.minimum_mass.max(1.),
        );
        for (p, t) in theta.iter().enumerate() {
            let cost = charge(&[*t]);
            let limit = self.budgets[p] * t[5];
            error = error
                .max(-t[5] / self.minimum_mass.max(1.))
                .max((cost - limit) / cost.max(limit.abs()).max(1.));
            for a in &self.anchors[p] {
                error = error.max(
                    project::dot(a, t).abs()
                        / (a.iter().map(|v| v.abs()).sum::<f64>().max(1.) * t[5].abs().max(1.)),
                );
            }
        }
        error.max(0.)
    }

    /// Bounded Dykstra projection. Returning an error retains a failed numerical
    /// solve; reaching the iteration cap is never called a feasible update.
    pub fn project(
        &self,
        input: &[Parameters],
        max_cycles: usize,
        tolerance: f64,
    ) -> Result<Projection> {
        self.validate()?;
        if input.len() != self.budgets.len()
            || input
                .iter()
                .flatten()
                .any(|x| !x.is_finite() || x.abs() > 1e15)
            || !(1..=20000).contains(&max_cycles)
            || !tolerance.is_finite()
            || !(1e-12..=1e-7).contains(&tolerance)
        {
            return Err("invalid bounded joint projection request".into());
        }
        let mut x = input.to_vec();
        let mut corrections = vec![vec![[0.; 6]; x.len()]; 4];
        for cycle in 1..=max_cycles {
            let mut movement: f64 = 0.;
            for (kind, correction) in corrections.iter_mut().enumerate() {
                let old = x.clone();
                for (t, d) in x.iter_mut().zip(correction.iter()) {
                    for j in 0..6 {
                        t[j] += d[j];
                    }
                }
                let before = x.clone();
                match kind {
                    0 => {
                        for (t, b) in x.iter_mut().zip(&self.basis) {
                            project::linear(t, b);
                        }
                    }
                    1 => {
                        for (t, b) in x.iter_mut().zip(&self.budgets) {
                            *t = project::perspective(*t, *b)?;
                        }
                    }
                    2 => {
                        let shift = ((self.minimum_mass - x.iter().map(|t| t[5]).sum::<f64>())
                            / x.len() as f64)
                            .max(0.);
                        for t in &mut x {
                            t[5] += shift;
                        }
                    }
                    _ => {
                        let values: Vec<_> = x.iter().flatten().copied().collect();
                        let weights: Vec<_> = x.iter().flat_map(|_| COSTS).collect();
                        let flat = project::weighted_l1(&values, &weights, self.radius)?;
                        for (t, src) in x.iter_mut().zip(flat.as_chunks::<6>().0) {
                            t.copy_from_slice(src);
                        }
                    }
                }
                for p in 0..x.len() {
                    for j in 0..6 {
                        correction[p][j] = before[p][j] - x[p][j];
                        movement = movement.max((old[p][j] - x[p][j]).abs());
                    }
                }
            }
            let scale = x.iter().flatten().map(|v| v.abs()).fold(1., f64::max);
            let violation = self.violation(&x);
            if violation <= tolerance && movement / scale <= tolerance {
                let squared_distance = input
                    .iter()
                    .flatten()
                    .zip(x.iter().flatten())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                return Ok(Projection {
                    theta: x,
                    cycles: cycle,
                    normalized_violation: violation,
                    normalized_step: movement / scale,
                    squared_distance,
                });
            }
        }
        Err("joint projection did not converge within its declared cycle budget".into())
    }
}

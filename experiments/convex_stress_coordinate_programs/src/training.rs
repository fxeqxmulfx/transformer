//! Ordinary CE on all CM57 joint coordinates. Observed constraints are fixed
//! before optimization. Forward mode may project onto those constraints but
//! performs no gradient step; no heldout label enters fitting or projection.
use crate::{
    constraints::{Domain, Projection},
    data::Record,
    head::Head,
    *,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Mode {
    Forward,
    Full,
    Mini,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Policy {
    pub mode: Mode,
    pub epochs: usize,
    pub batch: usize,
    pub learning_rate: f64,
    pub initial_total_mass: f64,
    pub projection_cycles: usize,
    pub projection_tolerance: f64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Forecast {
    pub scalar_work: u64,
    pub buffer_bytes: u64,
}

impl Policy {
    pub fn forecast(&self, n: usize, p: usize, v: usize) -> Result<Forecast> {
        let bytes = data::forecast(n, p, v)? as u128;
        if self.epochs > 128
            || (self.mode == Mode::Forward) != (self.epochs == 0)
            || !(1..=4096).contains(&self.batch)
            || !(1..=20000).contains(&self.projection_cycles)
            || !self.projection_tolerance.is_finite()
            || !(1e-12..=1e-7).contains(&self.projection_tolerance)
            || !self.learning_rate.is_finite()
            || !(1e-12..=1e12).contains(&self.learning_rate)
            || !self.initial_total_mass.is_finite()
            || !(1e-6..=1e12).contains(&self.initial_total_mass)
        {
            return Err("invalid bounded joint-coordinate policy".into());
        }
        let batches = if self.mode == Mode::Mini {
            n.div_ceil(self.batch)
        } else {
            1
        };
        let steps = batches as u128 * self.epochs as u128;
        let (n, p, v) = (n as u128, p as u128, v as u128);
        let scalar = (4 * n + 8 * n * self.epochs as u128) * p * (256 * v + 512)
            + (steps + 1)
                * (2048 * n * p + self.projection_cycles as u128 * (16384 * p + 64 * n * p));
        let bytes = bytes + 64 * (steps * (128 + 128 * p) + 16 * n * self.epochs as u128);
        if scalar > 600_000_000_000_000 || bytes > MAX_BYTES {
            return Err("joint-coordinate work or trace storage forecast refused".into());
        }
        Ok(Forecast {
            scalar_work: scalar as u64,
            buffer_bytes: bytes as u64,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Step {
    pub epoch: usize,
    pub record_ids: Vec<usize>,
    pub before_ce: f64,
    pub after_ce: f64,
    pub gradient: Vec<Parameters>,
    pub projection: Projection,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Work {
    pub gradient_batches: usize,
    pub gradient_records: usize,
    pub effective_updates: usize,
    pub projection_cycles: usize,
    pub complete_logit_products: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Outcome {
    pub policy: Policy,
    pub forecast: Forecast,
    pub domain: Domain,
    pub initial_projection: Projection,
    pub initial_ce: f64,
    pub final_ce: f64,
    pub steps: Vec<Step>,
    pub work: Work,
    pub head: Head,
}

pub fn score(head: &Head, rows: &[Record]) -> Result<f64> {
    if rows.is_empty() {
        return Err("empty score batch".into());
    }
    let mut loss = 0.;
    for r in rows {
        if r.label >= head.coordinates.len() {
            return Err("batch label outside vocabulary".into());
        }
        loss += ce(&head.logits(&r.features)?, r.label);
    }
    Ok(loss / rows.len() as f64)
}

/// Differentiate the CM57 affine product identity. Finite differences and
/// independent full complex products validate this reduction in the tests.
pub fn gradient(head: &Head, rows: &[Record]) -> Result<(f64, Vec<Parameters>)> {
    if rows.is_empty() {
        return Err("empty gradient batch".into());
    }
    let mut gradient = vec![[0.; 6]; head.theta.len()];
    let mut loss = 0.;
    for r in rows {
        if r.label >= head.coordinates.len() {
            return Err("gradient label outside vocabulary".into());
        }
        let z = head.logits(&r.features)?;
        loss += ce(&z, r.label);
        let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut residual: Vec<_> = z.iter().map(|x| (x - max).exp()).collect();
        let mass: f64 = residual.iter().sum();
        for x in &mut residual {
            *x /= mass;
        }
        residual[r.label] -= 1.;
        let e1: f64 = residual
            .iter()
            .zip(&head.coordinates)
            .map(|(r, e)| r * 2. * e)
            .sum();
        let e2: f64 = residual
            .iter()
            .zip(&head.coordinates)
            .map(|(r, e)| -r * e * e)
            .sum();
        let norm = head::normalizer(&r.features) * rows.len() as f64;
        for (g, h) in gradient.iter_mut().zip(&r.features) {
            for j in 0..5 {
                g[j] += e1 * h[j] / norm;
            }
            g[5] += e2 / norm;
        }
    }
    Ok((loss / rows.len() as f64, gradient))
}

pub fn train(
    rows: &[Record],
    coordinates: Vec<f64>,
    budgets: Vec<f64>,
    radius: f64,
    minimum_mass: f64,
    policy: Policy,
) -> Result<Outcome> {
    let forecast = policy.forecast(rows.len(), budgets.len(), coordinates.len())?;
    let domain = Domain::new(rows, &coordinates, budgets, radius, minimum_mass)?;
    let mut initial = vec![[0.; 6]; domain.budgets.len()];
    let initial_mass = policy.initial_total_mass / initial.len() as f64;
    for p in &mut initial {
        p[5] = initial_mass;
    }
    let initial_projection = domain.project(
        &initial,
        policy.projection_cycles,
        policy.projection_tolerance,
    )?;
    let mut head = Head::new(coordinates, initial_projection.theta.clone())?;
    let initial_ce = score(&head, rows)?;
    let mut steps = vec![];
    let mut work = Work {
        projection_cycles: initial_projection.cycles,
        ..Work::default()
    };
    for epoch in 0..policy.epochs {
        let size = if policy.mode == Mode::Mini {
            policy.batch
        } else {
            rows.len()
        };
        for batch in rows.chunks(size) {
            let (before_ce, gradient) = gradient(&head, batch)?;
            let mut proposal = head.theta.clone();
            for (t, g) in proposal.iter_mut().zip(&gradient) {
                for j in 0..6 {
                    t[j] -= policy.learning_rate * g[j];
                }
            }
            let projection = domain.project(
                &proposal,
                policy.projection_cycles,
                policy.projection_tolerance,
            )?;
            work.effective_updates += usize::from(projection.theta != head.theta);
            work.projection_cycles += projection.cycles;
            head = Head::new(head.coordinates, projection.theta.clone())?;
            let after_ce = score(&head, batch)?;
            if !before_ce.is_finite() || !after_ce.is_finite() {
                return Err("nonfinite ordinary CE".into());
            }
            work.gradient_batches += 1;
            work.gradient_records += batch.len();
            steps.push(Step {
                epoch,
                record_ids: batch.iter().map(|r| r.id).collect(),
                before_ce,
                after_ce,
                gradient,
                projection,
            });
        }
    }
    let final_ce = score(&head, rows)?;
    work.complete_logit_products = 16
        * head.theta.len() as u64
        * head.coordinates.len() as u64
        * (2 * rows.len() as u64 + 2 * work.gradient_records as u64);
    Ok(Outcome {
        policy,
        forecast,
        domain,
        initial_projection,
        initial_ce,
        final_ce,
        steps,
        work,
        head,
    })
}

pub fn replay(saved: &Outcome, rows: &[Record]) -> Result<()> {
    saved.head.validate()?;
    let rebuilt = train(
        rows,
        saved.head.coordinates.clone(),
        saved.domain.budgets.clone(),
        saved.domain.radius,
        saved.domain.minimum_mass,
        saved.policy.clone(),
    )?;
    if *saved != rebuilt {
        return Err("joint training trace differs from deterministic replay".into());
    }
    Ok(())
}

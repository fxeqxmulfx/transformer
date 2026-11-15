//! CM66 weighted projection, corresponding to the original full Euclidean
//! objective for arbitrary proposals. Numerical KKT checks independently gate
//! acceptance; neither they nor the root finder are a Lean arithmetic refinement.
use crate::{domain::Domain, *};
mod solver;
use solver::Plan;

#[derive(Clone, Debug, PartialEq)]
pub struct Target {
    pub linear: Vec<f64>,
    pub free: Vec<f64>,
}
impl Target {
    pub fn validate(&self, domain: &Domain) -> Result<()> {
        if self.linear.len() != domain.shape.programs
            || self.free.len() != domain.parameters() - domain.shape.programs
            || self
                .linear
                .iter()
                .chain(&self.free)
                .any(|x| !x.is_finite() || x.abs() > 1e15)
        {
            return Err("invalid reduced weighted proposal".into());
        }
        Ok(())
    }
    pub fn step(
        domain: &Domain,
        q: &Parameters,
        gradient: &Parameters,
        rate: f64,
        work: &mut Work,
    ) -> Result<Self> {
        q.validate(domain)?;
        gradient.validate(domain)?;
        if !rate.is_finite() || !(1e-6..=1e6).contains(&rate) {
            return Err("invalid gradient rate".into());
        }
        work.charge(12 * domain.parameters() as u128)?;
        let p = domain.shape.programs;
        let target = Self {
            linear: (0..p)
                .map(|j| (1 + domain.known[j]) as f64 * q.values[j] - rate * gradient.values[j])
                .collect(),
            free: q.values[p..]
                .iter()
                .zip(&gradient.values[p..])
                .map(|(q, g)| q - rate * g)
                .collect(),
        };
        target.validate(domain)?;
        Ok(target)
    }
    pub fn objective(&self, domain: &Domain, q: &Parameters) -> Result<f64> {
        self.validate(domain)?;
        q.validate(domain)?;
        let p = domain.shape.programs;
        let masses: f64 = (0..p)
            .map(|j| {
                let k = (1 + domain.known[j]) as f64;
                k * (q.values[j] - self.linear[j] / k).powi(2)
            })
            .sum();
        Ok(masses
            + q.values[p..]
                .iter()
                .zip(&self.free)
                .map(|(q, v)| (q - v).powi(2))
                .sum::<f64>())
    }
}
#[derive(Clone, Debug, PartialEq)]
pub struct Projection {
    pub parameters: Parameters,
    pub multiplier: f64,
    pub maximum_feasibility_residual: f64,
    pub maximum_kkt_residual: f64,
    pub reduced_objective: f64,
    pub uniform_unknown_programs: usize,
}
pub fn project(domain: &Domain, target: &Target, work: &mut Work) -> Result<Projection> {
    domain.validate(work)?;
    target.validate(domain)?;
    let plans = Plan::prepare(domain, target, work)?;
    let masses = |lambda: f64, work: &mut Work| -> Result<Vec<f64>> {
        plans.iter().map(|p| p.mass(lambda, work)).collect()
    };
    let first = masses(0., work)?;
    let total: f64 = first.iter().sum();
    let boundary = total.clamp(domain.minimum_mass, domain.maximum_mass);
    let (lambda, weights) = if total == boundary {
        (0., first)
    } else {
        let (mut lo, mut hi) = if total > boundary {
            (0., 1.)
        } else {
            (-1., 0.)
        };
        for _ in 0..64 {
            if masses(lo, work)?.iter().sum::<f64>() >= boundary
                && masses(hi, work)?.iter().sum::<f64>() <= boundary
            {
                break;
            }
            if lo < 0. {
                lo *= 2.;
            } else {
                hi *= 2.;
            }
        }
        if masses(lo, work)?.iter().sum::<f64>() < boundary
            || masses(hi, work)?.iter().sum::<f64>() > boundary
        {
            return Err("weighted projection multiplier could not be bracketed".into());
        }
        for _ in 0..80 {
            let mid = (lo + hi) / 2.;
            if masses(mid, work)?.iter().sum::<f64>() > boundary {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let lambda = if total > boundary { hi } else { lo };
        (lambda, masses(lambda, work)?)
    };
    work.charge(16 * domain.parameters() as u128)?;
    let sh = domain.shape;
    let mut values = vec![0.; domain.parameters()];
    values[..sh.programs].copy_from_slice(&weights);
    for (p, plan) in plans.iter().enumerate() {
        for row in &plan.unknown {
            let theta = if row.uniform {
                0.
            } else {
                row.threshold(weights[p], work)?
            };
            for c in 0..sh.classes {
                values[sh.programs + row.offset + c] = if row.uniform {
                    weights[p] / sh.classes as f64
                } else {
                    (target.free[row.offset + c] - theta).max(0.)
                };
            }
        }
    }
    let parameters = Parameters { values };
    let feasibility = domain.check(&parameters, 1e-8, work)?;
    let kkt = audit(domain, target, &parameters, lambda, work)?;
    if !kkt.is_finite() || kkt > 1e-7 {
        return Err(format!("weighted KKT residual {kkt}").into());
    }
    let objective = target.objective(domain, &parameters)?;
    if !objective.is_finite() {
        return Err("nonfinite weighted projection objective".into());
    }
    Ok(Projection {
        parameters,
        multiplier: lambda,
        maximum_feasibility_residual: feasibility,
        maximum_kkt_residual: kkt,
        reduced_objective: objective,
        uniform_unknown_programs: plans.iter().filter(|p| p.uniform).count(),
    })
}

fn audit(
    domain: &Domain,
    target: &Target,
    q: &Parameters,
    lambda: f64,
    work: &mut Work,
) -> Result<f64> {
    let sh = domain.shape;
    work.charge(32 * domain.parameters() as u128)?;
    let total: f64 = q.values[..sh.programs].iter().sum();
    let mut residual = if lambda > 0. {
        (total - domain.maximum_mass).abs()
    } else if lambda < 0. {
        (total - domain.minimum_mass).abs()
    } else {
        0.
    };
    let mut derivatives: Vec<_> = (0..sh.programs)
        .map(|p| (1 + domain.known[p]) as f64 * q.values[p] - target.linear[p])
        .collect();
    for &(p, s) in &domain.unknown {
        let at = domain.offsets[p * sh.states + s];
        let offset = at - sh.programs;
        let (mut count, mut sum, mut maximum) = (0, 0., f64::NEG_INFINITY);
        for c in 0..sh.classes {
            maximum = maximum.max(target.free[offset + c]);
            if q.values[at + c] > 0. {
                count += 1;
                sum += target.free[offset + c] - q.values[at + c];
            }
        }
        let theta = if count > 0 {
            sum / count as f64
        } else {
            maximum
        };
        derivatives[p] -= theta;
        for c in 0..sh.classes {
            residual =
                residual.max((q.values[at + c] - (target.free[offset + c] - theta).max(0.)).abs());
        }
    }
    for (p, &d) in derivatives.iter().enumerate() {
        residual = residual.max(if q.values[p] > 0. {
            (d + lambda).abs()
        } else {
            (-(d + lambda)).max(0.)
        });
    }
    Ok(residual)
}

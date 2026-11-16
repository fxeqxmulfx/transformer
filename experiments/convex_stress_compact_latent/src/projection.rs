//! CM66/70 complete weighted projection, ported from reduced_latent at
//! b3aa191c. All numerical root iterations, coordinates and KKT checks remain.
//! The local limit includes plans, output and simultaneous temporary vectors;
//! input tensors and the dictionary belong to the caller.
use crate::domain::Domain;
use crate::{
    memory,
    rows::{Builder, Rows},
    *,
};
mod solver;
use solver::Plan;
mod target;
pub use target::Target;
#[cfg(test)]
mod tests;

#[derive(Debug)]
pub struct Projection {
    pub parameters: Parameters,
    pub multiplier: f64,
    pub maximum_feasibility_residual: f64,
    pub maximum_kkt_residual: f64,
    pub reduced_objective: f64,
    pub uniform_unknown_programs: usize,
}
pub fn project(
    domain: &Domain,
    target: &Target,
    limit: u64,
    work: &mut Work,
) -> Result<Projection> {
    domain::validate(domain, work)?;
    target.validate(domain)?;
    // Three multiplier vectors may coexist with the final mass vector and
    // KKT scratch. The row builder separately charges its full class scratch.
    let scratch = 4 * memory::vector_bytes::<f64>(domain.shape.programs) + 4096;
    let plans = Plan::prepare(domain, target, memory::remaining(limit, scratch)?, work)?;
    let row_room = memory::remaining(limit, scratch + Plan::allocation_bytes(&plans))?;
    let masses = |lambda: f64, work: &mut Work| -> Result<Vec<f64>> {
        let mut values = memory::vector(plans.len(), limit)?;
        for p in &plans {
            values.push(p.mass(lambda, work)?);
        }
        Ok(values)
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
    let mut values = Builder::new(sh.classes, domain.unknown.len(), row_room)?;
    let mut row_values = memory::vector(sh.classes, row_room)?;
    for (p, plan) in plans.iter().enumerate() {
        for row in &plan.unknown {
            let theta = if row.uniform {
                0.
            } else {
                row.threshold(weights[p], work)?
            };
            row_values.clear();
            for c in 0..sh.classes {
                row_values.push(if row.uniform {
                    weights[p] / sh.classes as f64
                } else {
                    (target.free.get(row.offset / sh.classes, c).unwrap() - theta).max(0.)
                });
            }
            values.push(&row_values)?;
        }
    }
    let parameters = Parameters::from_parts(domain, weights, values.finish()?, limit)?;
    let feasibility = parameters.check(domain, 1e-8, work)?;
    let kkt = audit(domain, target, &parameters, lambda, limit, work)?;
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
    limit: u64,
    work: &mut Work,
) -> Result<f64> {
    let sh = domain.shape;
    work.charge(32 * domain.parameters() as u128)?;
    let total: f64 = q.masses().iter().sum();
    let mut residual = if lambda > 0. {
        (total - domain.maximum_mass).abs()
    } else if lambda < 0. {
        (total - domain.minimum_mass).abs()
    } else {
        0.
    };
    let mut derivatives = memory::vector(sh.programs, limit)?;
    derivatives.extend(
        (0..sh.programs).map(|p| (1 + domain.known[p]) as f64 * q.masses()[p] - target.linear[p]),
    );
    for &(p, s) in &domain.unknown {
        let at = domain.offsets[p * sh.states + s];
        let row = (at - sh.programs) / sh.classes;
        let (mut count, mut sum, mut maximum) = (0, 0., f64::NEG_INFINITY);
        for c in 0..sh.classes {
            let target_value = *target.free.get(row, c).unwrap();
            let value = *q.unknown_rows().get(row, c).unwrap();
            maximum = maximum.max(target_value);
            if value > 0. {
                count += 1;
                sum += target_value - value;
            }
        }
        let theta = if count > 0 {
            sum / count as f64
        } else {
            maximum
        };
        derivatives[p] -= theta;
        for c in 0..sh.classes {
            residual = residual.max(
                (q.unknown_rows().get(row, c).unwrap()
                    - (target.free.get(row, c).unwrap() - theta).max(0.))
                .abs(),
            );
        }
    }
    for (p, &d) in derivatives.iter().enumerate() {
        residual = residual.max(if q.masses()[p] > 0. {
            (d + lambda).abs()
        } else {
            (-(d + lambda)).max(0.)
        });
    }
    Ok(residual)
}

//! Numerical Euclidean projection onto CM64's full joint domain. Eliminate
//! known-row equalities, solve each unknown simplex row at a shared mass,
//! then solve the single global mass multiplier. Independent KKT checks gate
//! acceptance. The algorithm itself is not a claimed Lean refinement.
use crate::{domain::Domain, *};
mod audit;
mod solver;
use solver::Plan;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Projection {
    pub parameters: Parameters,
    pub multiplier: f64,
    pub maximum_feasibility_residual: f64,
    pub maximum_kkt_residual: f64,
    pub squared_distance: f64,
    pub uniform_unknown_programs: usize,
    pub exact_real_certificate: bool,
}

pub fn project(domain: &Domain, target: &Parameters, work: &mut Work) -> Result<Projection> {
    domain.validate()?;
    target.validate(domain.shape)?;
    let sh = domain.shape;
    let plans = Plan::prepare(domain, target, work)?;
    let masses = |lambda: f64, work: &mut Work| -> Result<Vec<f64>> {
        plans.iter().map(|p| p.mass(lambda, work)).collect()
    };
    let first = masses(0., work)?;
    let total = first.iter().sum::<f64>();
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
            return Err("joint projection multiplier could not be bracketed".into());
        }
        for _ in 0..80 {
            let mid = (lo + hi) / 2.;
            if masses(mid, work)?.iter().sum::<f64>() > boundary {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        // Choose the endpoint consistent with the active one-sided bound.
        let lambda = if total > boundary { hi } else { lo };
        (lambda, masses(lambda, work)?)
    };
    work.charge(8 * sh.parameters() as u128)?;
    let mut values = vec![0.; sh.parameters()];
    values[..sh.programs].copy_from_slice(&weights);
    for (p, plan) in plans.iter().enumerate() {
        let mass = weights[p];
        let mut unknown = 0;
        for s in 0..sh.states {
            if let Some(c) = domain.labels[p * sh.states + s] {
                values[sh.at(p, s, c)] = mass;
            } else {
                let row = &plan.unknown[unknown];
                unknown += 1;
                let theta = if row.uniform {
                    0.
                } else {
                    row.threshold(mass, work)?
                };
                for c in 0..sh.classes {
                    values[sh.at(p, s, c)] = if row.uniform {
                        mass / sh.classes as f64
                    } else {
                        (target.values[sh.at(p, s, c)] - theta).max(0.)
                    };
                }
            }
        }
    }
    let parameters = Parameters { values };
    let feasibility = domain.check(&parameters, 1e-8)?;
    let kkt = audit::check(domain, target, &parameters, lambda, work)?;
    if !kkt.is_finite() || kkt > 1e-7 {
        return Err(format!("joint projection KKT residual{kkt} exceeds1e-7").into());
    }
    let squared_distance = parameters
        .values
        .iter()
        .zip(&target.values)
        .map(|(a, b)| (a - b).powi(2))
        .sum();
    Ok(Projection {
        parameters,
        multiplier: lambda,
        maximum_feasibility_residual: feasibility,
        maximum_kkt_residual: kkt,
        squared_distance,
        uniform_unknown_programs: plans.iter().filter(|p| p.uniform).count(),
        exact_real_certificate: false,
    })
}

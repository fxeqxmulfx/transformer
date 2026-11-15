//! Independent first-order projection checks from the saved full tensors.
//! Reconstruct simplex multipliers from positive output entries, not from
//! the sorting/root-finder's internal state. No optimum is assumed as input.
use super::*;

pub(super) fn check(
    domain: &Domain,
    target: &Parameters,
    q: &Parameters,
    lambda: f64,
    work: &mut Work,
) -> Result<f64> {
    let sh = domain.shape;
    work.charge(24 * sh.parameters() as u128)?;
    let total = q.values[..sh.programs].iter().sum::<f64>();
    let mut residual = if lambda > 0. {
        (total - domain.maximum_mass).abs()
    } else if lambda < 0. {
        (total - domain.minimum_mass).abs()
    } else {
        0.
    };
    for p in 0..sh.programs {
        let mass = q.values[p];
        let mut derivative = mass - target.values[p];
        for s in 0..sh.states {
            if let Some(y) = domain.labels[p * sh.states + s] {
                derivative += mass - target.values[sh.at(p, s, y)];
            } else {
                let mut active = 0;
                let mut sum = 0.;
                let mut maximum = f64::NEG_INFINITY;
                for c in 0..sh.classes {
                    let i = sh.at(p, s, c);
                    maximum = maximum.max(target.values[i]);
                    if q.values[i] > 0. {
                        active += 1;
                        sum += target.values[i] - q.values[i];
                    }
                }
                let theta = if active > 0 {
                    sum / active as f64
                } else {
                    maximum
                };
                derivative -= theta;
                for c in 0..sh.classes {
                    let i = sh.at(p, s, c);
                    residual =
                        residual.max((q.values[i] - (target.values[i] - theta).max(0.)).abs());
                }
            }
        }
        residual = residual.max(if mass > 0. {
            (derivative + lambda).abs()
        } else {
            (-(derivative + lambda)).max(0.)
        });
    }
    Ok(residual)
}

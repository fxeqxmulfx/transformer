//! CM104 normalization/reconstruction, preserving every learned state function.
//! The pre-existing zero component represents unused mass. Its state and rows
//! remain structurally zero; all other components remain stored at zero weight.
use crate::{C, Result};
use rotary::{atom::State, model::Model};
pub type Table = Vec<Vec<State>>;

pub fn zero(model: &Model) -> Table {
    vec![vec![[C::default(); 2]; model.objects]; model.atoms.len()]
}
pub fn weighted(model: &Model) -> Table {
    model
        .atoms
        .iter()
        .zip(&model.weights)
        .map(|(a, &w)| a.output.iter().map(|u| [u[0] * w, u[1] * w]).collect())
        .collect()
}
pub fn masses(model: &Model, rows: &Table) -> Result<Vec<f64>> {
    if rows.len() != model.atoms.len() || rows.iter().any(|r| r.len() != model.objects) {
        return Err("incomplete weighted table".into());
    }
    rows.iter()
        .map(|r| {
            let mut mass = 0f64;
            for u in r {
                let mut size = 0.;
                for z in u {
                    let n = z.norm();
                    if !n.is_finite() || (n == 0. && *z != C::default()) {
                        return Err("nonfinite or underflowed full weighted row".into());
                    }
                    size += n;
                }
                if !size.is_finite() {
                    return Err("nonfinite row mass".into());
                }
                mass = mass.max(size);
            }
            Ok(mass)
        })
        .collect()
}

/// Exact CM104 formula in floating arithmetic. Existing native validation
/// tolerances remain explicit; no small nonzero mass is removed or thresholded.
pub fn reconstruct(model: &Model, rows: &Table) -> Result<Model> {
    model.validate()?;
    let mass = masses(model, rows)?;
    if mass[0] != 0. {
        return Err("structural unused-capacity rows changed".into());
    }
    let used: f64 = mass.iter().sum();
    if !used.is_finite() || used > model.budget + 2e-12 * model.budget {
        return Err("weighted table exceeds shared capacity".into());
    }
    let mut result = model.clone();
    for (j, (atom, r)) in result.atoms.iter_mut().zip(rows).enumerate() {
        result.weights[j] = mass[j];
        for (out, u) in atom.output.iter_mut().zip(r) {
            *out = if mass[j] == 0. {
                [C::default(); 2]
            } else {
                [u[0] / mass[j], u[1] / mass[j]]
            };
        }
    }
    result.weights[0] = (model.budget - used).max(0.);
    result.validate()?;
    Ok(result)
}

/// CM104 convex interpolation of weighted rows; multiplying independently
/// interpolated weights and normalized rows would give different logits.
pub fn mix(base: &Model, target: &Model, eta: f64) -> Result<Model> {
    base.validate()?;
    target.validate()?;
    if !eta.is_finite()
        || !(0. ..=1.).contains(&eta)
        || base.objects != target.objects
        || base.budget != target.budget
        || base.family != target.family
        || readouts::source::content_hash(base)? != readouts::source::content_hash(target)?
    {
        return Err("incompatible complete capacity interpolation".into());
    }
    if eta == 0. {
        return Ok(base.clone());
    }
    if eta == 1. {
        return Ok(target.clone());
    }
    let mut rows = weighted(base);
    let other = weighted(target);
    for (a, b) in rows.iter_mut().zip(&other) {
        for (u, v) in a.iter_mut().zip(b) {
            for k in 0..2 {
                u[k] = u[k] * (1. - eta) + v[k] * eta;
            }
        }
    }
    reconstruct(base, &rows)
}

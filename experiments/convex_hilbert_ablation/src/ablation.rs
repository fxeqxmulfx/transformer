//! Constant-state and zero-contribution constructions in the existing bounded family.
//! Feasibility is checked on the actual full geometry; no class is removed.
use crate::{C, Result};
use bilinear::train::Record;
use hilbert::model::Model;
use rotary::atom::State;
use serde::{Deserialize, Serialize};
#[derive(Debug, Serialize, Deserialize)]
pub struct Mean {
    pub state: State,
    pub records: usize,
    pub state_norm: f64,
    pub minimum_original_norm: f64,
    pub mean_original_norm: f64,
    pub mean_squared_euclidean_deviation: f64,
}
pub fn training_mean(model: &Model, records: &[Record<'_>]) -> Result<Mean> {
    if records.is_empty() {
        return Err("empty original training mean".into());
    }
    model.validate()?;
    let mut mean = [C::default(); 2];
    let mut minimum = f64::INFINITY;
    let mut mean_norm = 0.;
    for r in records {
        // The target value is deliberately never inspected here.
        let state = model.block.atom.state(r.prefix)?;
        for (m, z) in mean.iter_mut().zip(state) {
            *m += z / records.len() as f64;
        }
        let n = model.block.geometry.state_norm(state);
        minimum = minimum.min(n);
        mean_norm += n / records.len() as f64;
    }
    let norm = model.block.geometry.state_norm(mean);
    if !norm.is_finite() || norm > 1. + 2e-12 {
        return Err("actual training-mean state exceeds its unit ball".into());
    }
    let mut deviation = 0.;
    for r in records {
        let state = model.block.atom.state(r.prefix)?;
        for (z, m) in state.iter().zip(mean) {
            deviation += (*z - m).norm_sqr() / records.len() as f64;
        }
    }
    Ok(Mean {
        state: mean,
        records: records.len(),
        state_norm: norm,
        minimum_original_norm: minimum,
        mean_original_norm: mean_norm,
        mean_squared_euclidean_deviation: deviation,
    })
}
pub fn make(model: &Model, mean: &Mean, kind: &str) -> Result<Model> {
    model.validate()?;
    let mut out = model.clone();
    match kind {
        "intact" => {}
        "zero-state" => out.block.atom.initial = [C::default(); 2],
        "training-mean-state" => {
            out.block.atom.initial = mean.state;
            out.block.atom.transition.fill([
                [C::new(1., 0.), C::default()],
                [C::default(), C::new(1., 0.)],
            ]);
        }
        "new-block-only" => {
            out.parent.weights.fill(0.);
            out.parent.weights[0] = out.parent.budget;
        }
        _ => return Err("undeclared ablation".into()),
    }
    out.validate()?;
    Ok(out)
}

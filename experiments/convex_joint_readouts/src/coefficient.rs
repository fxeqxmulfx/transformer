//! Actual complete CE coefficients of weighted rows: no old weight/B multiplier.
//! Forward mode uses the existing finite feasible signed-bias CE queries from
//! CM98/99, after the caller reserves their capacity at the current centre.
use crate::{
    Result,
    capacity::{self, Table},
};
use bilinear::{
    model::cross_entropy,
    train::{Mode, Record},
};
use rotary::{atom::State, model::Model};

pub struct Batch {
    pub sum: Table,
    pub states: Vec<Vec<State>>,
    pub logits: Vec<Vec<f64>>,
    pub mean_ce: f64,
    pub maximum_logit_error: f64,
}
pub fn batch(model: &Model, records: &[Record<'_>], mode: Mode, h: f64) -> Result<Batch> {
    if records.is_empty() {
        return Err("empty coefficient minibatch".into());
    }
    model.validate()?;
    if mode == Mode::ForwardMini && (!h.is_finite() || h <= 0. || model.weights[0] + 1e-12 < h) {
        return Err("forward CE queries lack reserved feasible mass".into());
    }
    let mut result = Batch {
        sum: capacity::zero(model),
        states: Vec::new(),
        logits: Vec::new(),
        mean_ce: 0.,
        maximum_logit_error: 0.,
    };
    for r in records {
        // Include zero-weight learned components: they may acquire capacity.
        let states = model
            .atoms
            .iter()
            .map(|a| a.state(r.prefix))
            .collect::<Result<Vec<_>>>()?;
        let scores = model.logits(r.prefix)?;
        let independent = readouts::update::logits(model, &states)?;
        for (&a, &b) in scores.iter().zip(&independent) {
            if !a.is_finite() || !b.is_finite() {
                return Err("nonfinite complete logit".into());
            }
            result.maximum_logit_error = result.maximum_logit_error.max((a - b).abs());
        }
        result.mean_ce += cross_entropy(&scores, r.target)?;
        let scalars = rotary::train::scalars(&scores, r.target, mode, h)?;
        for (rows, state) in result.sum.iter_mut().zip(&states) {
            for (a, &s) in rows.iter_mut().zip(&scalars) {
                for k in 0..2 {
                    a[k] += state[k] * s;
                }
            }
        }
        result.states.push(states);
        result.logits.push(scores);
    }
    result.mean_ce /= records.len() as f64;
    if result.maximum_logit_error > 2e-12 {
        return Err("actual coefficient logits disagree".into());
    }
    Ok(result)
}
pub fn mean(rows: &mut Table, n: usize) -> Result<()> {
    if n == 0 {
        return Err("empty coefficient mean".into());
    }
    for z in rows.iter_mut().flatten().flatten() {
        *z /= n as f64;
    }
    Ok(())
}

//! Port of joint::coefficient::batch at f5ae34ef, retaining individual signals.
//! Evaluates each finite-forward query once, including all original classes.
use crate::Result;
use bilinear::{
    model::cross_entropy,
    train::{Mode, Record},
};
use rotary::{atom::State, model::Model};

pub struct Batch {
    pub states: Vec<Vec<State>>,
    pub signals: Vec<Vec<f64>>,
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
        return Err("forward queries lack feasible reserve".into());
    }
    let mut b = Batch {
        states: vec![],
        signals: vec![],
        logits: vec![],
        mean_ce: 0.,
        maximum_logit_error: 0.,
    };
    for r in records {
        let states = model
            .atoms
            .iter()
            .map(|a| a.state(r.prefix))
            .collect::<Result<Vec<_>>>()?;
        let scores = model.logits(r.prefix)?;
        let independent = readouts::update::logits(model, &states)?;
        for (&x, &y) in scores.iter().zip(&independent) {
            if !x.is_finite() || !y.is_finite() {
                return Err("nonfinite actual logit".into());
            }
            b.maximum_logit_error = b.maximum_logit_error.max((x - y).abs());
        }
        b.mean_ce += cross_entropy(&scores, r.target)?;
        b.signals
            .push(rotary::train::scalars(&scores, r.target, mode, h)?);
        b.states.push(states);
        b.logits.push(scores);
    }
    b.mean_ce /= records.len() as f64;
    if b.maximum_logit_error > 2e-12 {
        return Err("actual coefficient logits disagree".into());
    }
    Ok(b)
}

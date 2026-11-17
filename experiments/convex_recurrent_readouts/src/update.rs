//! CM100 actual radial row update and complete conditional-CE interpolation.
//! Ordinary Mini uses the full readout gradient. Forward uses finite CE scalars
//! in the same contraction; neither mode backpropagates through recurrent states.
use crate::{C, Result};
use bilinear::{
    model::cross_entropy,
    train::{Mode, Record},
};
use rotary::{
    atom::{State, row},
    model::Model,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Config {
    pub mode: Mode,
    pub query_step: f64,
    pub learning_rate: f64,
    pub line_iterations: usize,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct Receipt {
    pub records: usize,
    pub centre_ce: f64,
    pub proposal_ce: f64,
    pub after_ce: f64,
    pub eta: f64,
    pub bias_loss_queries: usize,
    pub line_loss_queries: usize,
    pub centre_validation_and_check_ce_evaluations: usize,
    pub total_ce_evaluations: usize,
    pub row_coordinates_processed: usize,
    pub maximum_actual_logit_error: f64,
    pub maximum_operator_norm_excess: f64,
}

/// CM100 projectRow, retaining both complex coefficients as stored coordinates.
pub fn project_row(u: State) -> Result<State> {
    let n = u[0].norm() + u[1].norm();
    if !n.is_finite() {
        return Err("nonfinite complete readout row".into());
    }
    Ok(if n > 1. { [u[0] / n, u[1] / n] } else { u })
}

pub fn logits(model: &Model, states: &[State]) -> Result<Vec<f64>> {
    if states.len() != model.atoms.len() {
        return Err("incomplete component states".into());
    }
    let mut scores = vec![0.; model.objects];
    for ((atom, &w), &state) in model.atoms.iter().zip(&model.weights).zip(states) {
        for (score, &u) in scores.iter_mut().zip(&atom.output) {
            *score += w * row(u, state).re;
        }
    }
    Ok(scores)
}

// Port of rotary train::line_search at534a258f. CM100 proves this complete
// readout interpolation executes the same affine logits in every class.
fn line_search(
    base: &[Vec<f64>],
    candidate: &[Vec<f64>],
    records: &[Record<'_>],
    iterations: usize,
) -> Result<(f64, f64, f64)> {
    let evaluate = |eta: f64| -> Result<f64> {
        let mut loss = 0.;
        for ((x, y), r) in base.iter().zip(candidate).zip(records) {
            let scores: Vec<_> = x
                .iter()
                .zip(y)
                .map(|(&x, &y)| (1. - eta) * x + eta * y)
                .collect();
            loss += cross_entropy(&scores, r.target)?;
        }
        Ok(loss / records.len() as f64)
    };
    let (mut lo, mut hi) = (0., 1.);
    let ratio = (5f64.sqrt() - 1.) / 2.;
    let (mut x, mut y) = (hi - ratio * (hi - lo), lo + ratio * (hi - lo));
    let (mut fx, mut fy) = (evaluate(x)?, evaluate(y)?);
    let (mut best_eta, mut best_loss) = (0., evaluate(0.)?);
    let proposal = evaluate(1.)?;
    if proposal < best_loss {
        best_eta = 1.;
        best_loss = proposal;
    }
    for _ in 0..iterations {
        for (eta, loss) in [(x, fx), (y, fy)] {
            if loss < best_loss {
                best_eta = eta;
                best_loss = loss;
            }
        }
        if fx < fy {
            hi = y;
            y = x;
            fy = fx;
            x = hi - ratio * (hi - lo);
            fx = evaluate(x)?;
        } else {
            lo = x;
            x = y;
            fx = fy;
            y = lo + ratio * (hi - lo);
            fy = evaluate(y)?;
        }
    }
    Ok((best_eta, best_loss, proposal))
}

pub fn step(model: &mut Model, records: &[Record<'_>], config: Config) -> Result<Receipt> {
    if records.is_empty()
        || config.line_iterations == 0
        || !config.learning_rate.is_finite()
        || config.learning_rate <= 0.
    {
        return Err("invalid conditional readout update".into());
    }
    model.shrink_for_queries(config.query_step)?;
    let states: Vec<Vec<State>> = records
        .iter()
        .map(|r| {
            model
                .atoms
                .iter()
                .map(|a| a.state(r.prefix))
                .collect::<Result<_>>()
        })
        .collect::<Result<_>>()?;
    let base: Vec<_> = states
        .iter()
        .map(|s| logits(model, s))
        .collect::<Result<_>>()?;
    let mut signals = Vec::with_capacity(records.len());
    let mut centre_ce = 0.;
    for (r, scores) in records.iter().zip(&base) {
        centre_ce += cross_entropy(scores, r.target)?;
        signals.push(rotary::train::scalars(
            scores,
            r.target,
            config.mode,
            config.query_step,
        )?);
    }
    centre_ce /= records.len() as f64;
    let mut proposed = model.clone();
    for (j, atom) in proposed.atoms.iter_mut().enumerate() {
        let mut gradient = vec![[C::default(); 2]; model.objects];
        for (state, signal) in states.iter().zip(&signals) {
            for (grad, &scalar) in gradient.iter_mut().zip(signal) {
                for k in 0..2 {
                    grad[k] +=
                        state[j][k].conj() * (model.weights[j] * scalar / records.len() as f64);
                }
            }
        }
        for (out, grad) in atom.output.iter_mut().zip(gradient) {
            *out = project_row([
                out[0] - grad[0] * config.learning_rate,
                out[1] - grad[1] * config.learning_rate,
            ])?;
        }
    }
    proposed.validate()?;
    let candidate: Vec<_> = states
        .iter()
        .map(|s| logits(&proposed, s))
        .collect::<Result<_>>()?;
    let (eta, after_ce, proposal_ce) =
        line_search(&base, &candidate, records, config.line_iterations)?;
    for (atom, other) in model.atoms.iter_mut().zip(&proposed.atoms) {
        for (out, point) in atom.output.iter_mut().zip(&other.output) {
            for k in 0..2 {
                out[k] = out[k] * (1. - eta) + point[k] * eta;
            }
        }
    }
    let excess = model.validate()?;
    let mut error = 0f64;
    let mut actual_ce = 0.;
    for ((r, base), candidate) in records.iter().zip(&base).zip(&candidate) {
        let actual = model.logits(r.prefix)?;
        for ((&a, &b), &c) in actual.iter().zip(base).zip(candidate) {
            error = error.max((a - ((1. - eta) * b + eta * c)).abs());
        }
        actual_ce += cross_entropy(&actual, r.target)?;
    }
    if error > 2e-12 || (actual_ce / records.len() as f64 - after_ce).abs() > 2e-12 {
        return Err("complete conditional interpolation differs from actual model".into());
    }
    let bias = if config.mode == Mode::ForwardMini {
        2 * model.objects * records.len()
    } else {
        0
    };
    let line = (4 + config.line_iterations) * records.len();
    Ok(Receipt {
        records: records.len(),
        centre_ce,
        proposal_ce,
        after_ce,
        eta,
        bias_loss_queries: bias,
        line_loss_queries: line,
        centre_validation_and_check_ce_evaluations: 3 * records.len(),
        total_ce_evaluations: bias + line + 3 * records.len(),
        row_coordinates_processed: 4 * model.objects * model.atoms.len(),
        maximum_actual_logit_error: error,
        maximum_operator_norm_excess: excess,
    })
}

//! CM104 conditional capacity steps. No content coordinate is replaced by a
//! prescribed feature. This optimizer phase acts on already learned states.
use crate::{Result, capacity, coefficient, diagnostic};
use bilinear::{
    model::cross_entropy,
    train::{Mode, Record},
};
use rotary::model::Model;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Config {
    pub mode: Mode,
    pub query_step: f64,
    pub line_iterations: usize,
    pub step_ceiling: f64,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct Receipt {
    pub records: usize,
    pub centre_ce: f64,
    pub after_ce: f64,
    pub eta: f64,
    pub selected_component: usize,
    pub selected_weight_before: f64,
    pub selected_weight_after: f64,
    pub computed_signal_gap: f64,
    pub bias_loss_queries: usize,
    pub line_loss_queries: usize,
    pub total_ce_evaluations: usize,
    pub real_coordinates: usize,
    pub maximum_actual_logit_error: f64,
    pub maximum_operator_norm_excess: f64,
    pub shared_capacity_excess: f64,
}

// Ported from rotary::train::line_search at33511122. CM104 supplies affine
// actual logits for weighted-row interpolation, including redistributed mass.
fn line_search(
    base: &[Vec<f64>],
    target: &[Vec<f64>],
    records: &[Record<'_>],
    iterations: usize,
    ceiling: f64,
) -> Result<(f64, f64)> {
    let evaluate = |eta: f64| -> Result<f64> {
        let mut loss = 0.;
        for ((x, y), r) in base.iter().zip(target).zip(records) {
            let scores: Vec<_> = x
                .iter()
                .zip(y)
                .map(|(&x, &y)| (1. - eta) * x + eta * y)
                .collect();
            loss += cross_entropy(&scores, r.target)?;
        }
        Ok(loss / records.len() as f64)
    };
    let (mut lo, mut hi) = (0., ceiling);
    let ratio = (5f64.sqrt() - 1.) / 2.;
    let (mut x, mut y) = (hi - ratio * (hi - lo), lo + ratio * (hi - lo));
    let (mut fx, mut fy) = (evaluate(x)?, evaluate(y)?);
    let (mut best_eta, mut best_loss) = (0., evaluate(0.)?);
    let end = evaluate(ceiling)?;
    if end < best_loss {
        best_eta = ceiling;
        best_loss = end;
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
    Ok((best_eta, best_loss))
}

pub fn step(model: &mut Model, records: &[Record<'_>], config: Config) -> Result<Receipt> {
    if records.is_empty()
        || config.line_iterations == 0
        || !config.step_ceiling.is_finite()
        || config.step_ceiling <= 0.
        || config.step_ceiling > 1.
    {
        return Err("invalid joint-capacity update".into());
    }
    // Same explicit reserve in both modes; the receipt's centre is after it.
    model.shrink_for_queries(config.query_step)?;
    let mut b = coefficient::batch(model, records, config.mode, config.query_step)?;
    coefficient::mean(&mut b.sum, records.len())?;
    let (target, support) = diagnostic::direction(model, &b.sum)?;
    let candidate = b
        .states
        .iter()
        .map(|s| readouts::update::logits(&target, s))
        .collect::<Result<Vec<_>>>()?;
    let (eta, after_ce) = line_search(
        &b.logits,
        &candidate,
        records,
        config.line_iterations,
        config.step_ceiling,
    )?;
    let before = model.weights[support.selected];
    let next = capacity::mix(model, &target, eta)?;
    let mut error = b.maximum_logit_error;
    let mut actual_ce = 0.;
    for ((r, base), target) in records.iter().zip(&b.logits).zip(&candidate) {
        let actual = next.logits(r.prefix)?;
        for ((&x, &y), &z) in actual.iter().zip(base).zip(target) {
            error = error.max((x - ((1. - eta) * y + eta * z)).abs());
        }
        actual_ce += cross_entropy(&actual, r.target)?;
    }
    actual_ce /= records.len() as f64;
    if error > 2e-12 || (actual_ce - after_ce).abs() > 2e-12 || after_ce > b.mean_ce + 2e-12 {
        return Err("actual reconstructed CE disagrees with conditional step".into());
    }
    let norm_excess = next.validate()?;
    let capacity_excess = (capacity::masses(&next, &capacity::weighted(&next))?
        .iter()
        .sum::<f64>()
        - next.budget)
        .max(0.);
    *model = next;
    let bias = if config.mode == Mode::ForwardMini {
        2 * model.objects * records.len()
    } else {
        0
    };
    let line = (4 + config.line_iterations) * records.len();
    Ok(Receipt {
        records: records.len(),
        centre_ce: b.mean_ce,
        after_ce: actual_ce,
        eta,
        selected_component: support.selected,
        selected_weight_before: before,
        selected_weight_after: model.weights[support.selected],
        computed_signal_gap: support.approximate_gap,
        bias_loss_queries: bias,
        line_loss_queries: line,
        total_ce_evaluations: bias + line + 3 * records.len(),
        real_coordinates: 4 * model.atoms.len() * model.objects,
        maximum_actual_logit_error: error,
        maximum_operator_norm_excess: norm_excess,
        shared_capacity_excess: capacity_excess,
    })
}

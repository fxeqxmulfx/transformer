//! CM95 feasible ordinary-CE queries and convex mixture updates.
//! The component search has no recurrent backward pass in either mode.
use crate::{Result, model::Model, proposal::Search};
use baseline::random::Random;
use bilinear::{
    model::cross_entropy,
    train::{Mode, Record, forward_scalars},
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Config {
    pub mode: Mode,
    pub query_step: f64,
    pub sweeps: usize,
    pub line_iterations: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Receipt {
    pub records: usize,
    pub centre_ce: f64,
    pub after_ce: f64,
    pub bias_loss_queries: usize,
    pub component_loss_queries: usize,
    pub line_loss_queries: usize,
    pub eta: f64,
    pub component_trial_ce: f64,
    pub components_stored: usize,
    pub maximum_operator_norm_excess: f64,
    pub unused_mass: f64,
}

pub fn scalars(logits: &[f64], target: usize, mode: Mode, h: f64) -> Result<Vec<f64>> {
    cross_entropy(logits, target)?;
    match mode {
        Mode::ForwardMini => forward_scalars(logits, target, h),
        Mode::Mini => {
            let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let probabilities: Vec<_> = logits.iter().map(|x| (x - max).exp()).collect();
            let sum: f64 = probabilities.iter().sum();
            Ok(probabilities
                .into_iter()
                .enumerate()
                .map(|(c, p)| p / sum - f64::from(c == target))
                .collect())
        }
    }
}

/// Port of shared-atoms line_search at 41cffd66, without tensor assumptions.
pub(crate) fn line_search(
    base: &[Vec<f64>],
    candidate: &[Vec<f64>],
    records: &[Record<'_>],
    iterations: usize,
) -> Result<(f64, f64, usize)> {
    let evaluate = |eta: f64| -> Result<f64> {
        let mut total = 0.;
        for ((x, y), record) in base.iter().zip(candidate).zip(records) {
            let logits: Vec<_> = x
                .iter()
                .zip(y)
                .map(|(&x, &y)| (1. - eta) * x + eta * y)
                .collect();
            total += cross_entropy(&logits, record.target)?;
        }
        Ok(total / records.len() as f64)
    };
    let (mut lo, mut hi) = (0., 1.);
    let ratio = (5f64.sqrt() - 1.) / 2.;
    let (mut x, mut y) = (hi - ratio * (hi - lo), lo + ratio * (hi - lo));
    let (mut fx, mut fy) = (evaluate(x)?, evaluate(y)?);
    let (mut best_eta, mut best_loss) = (0., evaluate(0.)?);
    let fone = evaluate(1.)?;
    if fone < best_loss {
        best_eta = 1.;
        best_loss = fone;
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
    Ok((best_eta, best_loss, (4 + iterations) * records.len()))
}

pub fn step(
    model: &mut Model,
    records: &[Record<'_>],
    config: Config,
    rng: &mut Random,
) -> Result<Receipt> {
    if records.is_empty() || config.line_iterations == 0 || config.sweeps == 0 {
        return Err("empty batch or component/line search".into());
    }
    model.shrink_for_queries(config.query_step)?;
    let mut base = Vec::with_capacity(records.len());
    let mut signals = Vec::with_capacity(records.len());
    let mut centre_ce = 0.;
    for record in records {
        let logits = model.logits(record.prefix)?;
        centre_ce += cross_entropy(&logits, record.target)?;
        signals.push(scalars(
            &logits,
            record.target,
            config.mode,
            config.query_step,
        )?);
        base.push(logits);
    }
    centre_ce /= records.len() as f64;
    let mut search = Search {
        base: &base,
        signals: &signals,
        records,
        budget: model.budget,
        loss_queries: 0,
    };
    let previous = if model.atoms.len() > 1 {
        model.atoms.last()
    } else {
        None
    };
    let (atom, component_trial_ce) = search.propose(model.objects, config.sweeps, rng, previous)?;
    let candidate: Vec<Vec<f64>> = records
        .iter()
        .map(|r| {
            Ok(atom
                .complex(r.prefix)?
                .iter()
                .map(|z| z.re * model.budget)
                .collect())
        })
        .collect::<Result<_>>()?;
    let (eta, after_ce, line_loss_queries) =
        line_search(&base, &candidate, records, config.line_iterations)?;
    model.mix(atom, eta)?;
    Ok(Receipt {
        records: records.len(),
        centre_ce,
        after_ce,
        bias_loss_queries: if config.mode == Mode::ForwardMini {
            2 * model.objects * records.len()
        } else {
            0
        },
        component_loss_queries: search.loss_queries,
        line_loss_queries,
        eta,
        component_trial_ce,
        components_stored: model.atoms.len(),
        maximum_operator_norm_excess: model.validate()?,
        unused_mass: model.weights[0],
    })
}

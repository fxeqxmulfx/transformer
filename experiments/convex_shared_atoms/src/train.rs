//! CM92 convex component steps and feasible complete Mini/forward-Mini queries.
//! The running complete gradient and approximate component search are a native
//! optimization proposal, with no claimed global oracle or convergence theorem.
use crate::{Result, model::Model, oracle::Gradient};
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
    pub momentum: f64,
    pub restarts: usize,
    pub sweeps: usize,
    pub line_iterations: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Receipt {
    pub records: usize,
    pub centre_ce: f64,
    pub after_ce: f64,
    pub bias_loss_queries: usize,
    pub line_loss_queries: usize,
    pub eta: f64,
    pub approximate_component_value: f64,
    pub components_stored: usize,
    pub maximum_factor_energy_excess: f64,
    pub unused_mass: f64,
}

pub fn scalars(logits: &[f64], target: usize, mode: Mode, h: f64) -> Result<Vec<f64>> {
    if target >= logits.len() || logits.is_empty() {
        return Err("target outside original objects".into());
    }
    match mode {
        Mode::ForwardMini => forward_scalars(logits, target, h),
        Mode::Mini => {
            let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let p: Vec<_> = logits.iter().map(|x| (x - max).exp()).collect();
            let total: f64 = p.iter().sum();
            Ok(p.into_iter()
                .enumerate()
                .map(|(c, p)| p / total - f64::from(c == target))
                .collect())
        }
    }
}

/// Preserve every indexed record, original competitor, response multiplicity
/// and real/imaginary coordinate. G represents the exact pair (g,-g).
pub fn update_gradient(
    gradient: &mut Gradient,
    model: &Model,
    records: &[Record<'_>],
    config: Config,
) -> Result<(Vec<Vec<f64>>, f64)> {
    if records.is_empty()
        || gradient.shape != model.shape
        || !config.momentum.is_finite()
        || !(0. ..1.).contains(&config.momentum)
    {
        return Err("empty/incompatible batch or invalid momentum".into());
    }
    for value in &mut gradient.values {
        *value *= config.momentum;
    }
    let mut all_logits = Vec::with_capacity(records.len());
    let mut loss = 0.;
    for record in records {
        let input = model.shape.addresses(record.prefix)?;
        let logits: Vec<_> = model.complex(&input).into_iter().map(|x| x.re).collect();
        loss += cross_entropy(&logits, record.target)?;
        let derivative = scalars(&logits, record.target, config.mode, config.query_step)?;
        for &(d, count) in &input.columns {
            for &q in &input.queries {
                for (c, &scalar) in derivative.iter().enumerate() {
                    let index = gradient.index(c, q, d);
                    gradient.values[index] += (1. - config.momentum)
                        * scalar
                        * count as f64
                        * model.query_scale[q]
                        * model.context_scale[d]
                        / records.len() as f64;
                }
            }
        }
        all_logits.push(logits);
    }
    Ok((all_logits, loss / records.len() as f64))
}

fn line_search(
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
    gradient: &mut Gradient,
    records: &[Record<'_>],
    config: Config,
    rng: &mut Random,
) -> Result<Receipt> {
    if config.line_iterations == 0 {
        return Err("empty line search".into());
    }
    model.shrink_for_queries(config.query_step)?;
    let (base, centre_ce) = update_gradient(gradient, model, records, config)?;
    let (atom, value) =
        gradient.propose(rng, config.restarts, config.sweeps, model.atoms.last())?;
    let candidate: Vec<Vec<f64>> = records
        .iter()
        .map(|r| {
            Ok(atom
                .scaled_complex(
                    &model.shape.addresses(r.prefix)?,
                    &model.query_scale,
                    &model.context_scale,
                )
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
            2 * model.shape.objects * records.len()
        } else {
            0
        },
        line_loss_queries,
        eta,
        approximate_component_value: value,
        components_stored: model.atoms.len(),
        maximum_factor_energy_excess: model.validate()?,
        unused_mass: model.weights[0],
    })
}

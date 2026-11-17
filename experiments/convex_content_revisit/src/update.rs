//! Port of f5ae34ef joint update, selecting its direction with retained H.
//! The line search still measures only the current minibatch, not full CE.
use crate::{Result, cache::Cache};
use bilinear::{
    model::cross_entropy,
    train::{Mode, Record},
};
use joint::{
    capacity, diagnostic,
    update::{Config, Receipt},
};
use retained::line;
use rotary::model::Model;

pub fn step(
    model: &mut Model,
    cache: &mut Cache,
    ids: &[usize],
    records: &[Record<'_>],
    config: Config,
    version: usize,
) -> Result<Receipt> {
    if records.is_empty()
        || config.line_iterations == 0
        || !config.step_ceiling.is_finite()
        || config.step_ceiling <= 0.
        || config.step_ceiling > 1.
    {
        return Err("invalid retained update".into());
    }
    model.shrink_for_queries(config.query_step)?;
    let b = cache.observe(model, ids, records, config.mode, config.query_step, version)?;
    let (target, support) = diagnostic::direction(model, &cache.mean)?;
    let candidate = b
        .states
        .iter()
        .map(|s| readouts::update::logits(&target, s))
        .collect::<Result<Vec<_>>>()?;
    let (eta, after_ce) = line::line_search(
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
        return Err("actual reconstructed retained CE disagrees".into());
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

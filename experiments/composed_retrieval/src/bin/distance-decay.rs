//! Training-only CM118 intervention on the complete frozen CM116 Rewrite
//! cohort. Reload original objects and compare every baseline training logit.
#[path = "shared/training.rs"]
mod training;
use baseline::data::Example;
use composed_retrieval::{Result, data, model::Model, operator::Tables, train};
use serde_json::{Value, json};

const PRODUCER: &str = "9a222cfdd9c7364a864749f90dfc5d89d1d86a3f";
const GRID: [f64; 7] = [0., 0.0625, 0.125, 0.25, 0.5, 1., 2.];

fn intervention(m: &Model, rows: &[Example], raw: &[u8], after: &Value) -> Result<Value> {
    let layout = m.validate()?;
    let samples = train::samples(m, rows)?;
    let batch: Vec<_> = samples.iter().flatten().collect();
    let reach = batch
        .iter()
        .map(|s| s.prefix.len().saturating_sub(2))
        .max()
        .ok_or("empty supervision")?;
    let lengths = training::lengths(&batch, rows);
    let residual = &m.weights[layout.relation.clone()];
    let center = residual.len() / 2;
    let unseen = residual
        .iter()
        .enumerate()
        .filter(|(i, _)| i.abs_diff(center) > reach);
    if unseen.clone().any(|(_, &v)| v != 0.) {
        return Err("an unobservable residual moved from zero".into());
    }
    let mut peaks: Vec<_> = residual
        .iter()
        .copied()
        .enumerate()
        .filter(|(i, _)| i.abs_diff(center) <= reach)
        .map(|(i, v)| (i as i64 - center as i64, v))
        .collect();
    peaks.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    peaks.truncate(3);
    let tables = Tables::new(m, &lengths)?;
    let gradient = train::analytic(m, &tables, &batch);
    let slope = -gradient[layout.relation.clone()]
        .iter()
        .enumerate()
        .map(|(i, g)| i.abs_diff(center) as f64 * g)
        .sum::<f64>();
    let with_decay = |decay: f64| {
        let mut perturbed = m.clone();
        for (i, value) in perturbed.weights[layout.relation.clone()]
            .iter_mut()
            .enumerate()
        {
            *value -= decay * i.abs_diff(center) as f64;
        }
        perturbed
    };
    let step = 1. / 65536.;
    let loss = |a| -> Result<f64> {
        train::metrics(&with_decay(a), &samples)?["ce"]
            .as_f64()
            .ok_or_else(|| "missing CE".into())
    };
    let origin = after["ce"].as_f64().ok_or("receipt CE")?;
    let numerical = (4. * (loss(step)? - origin) - (loss(2. * step)? - origin)) / (2. * step);
    if (slope - numerical).abs() > 2e-7 * (1. + slope.abs()) {
        return Err(format!(
            "distance derivative differs from feasible queries: {slope} vs {numerical}"
        )
        .into());
    }
    let mut grid = Vec::new();
    for decay in GRID {
        let mut score = training::scores(
            &with_decay(decay),
            &lengths,
            rows,
            (decay == 0.).then_some((raw, after)),
        )?;
        score["decay"] = json!(decay);
        grid.push(score);
    }
    Ok(
        json!({"max_observed_distance":reach,"unobserved_residuals":unseen.count(),
        "peak_observed_residuals":peaks,"ce_right_derivative_at_zero":slope,"forward_queried_derivative":numerical,"training":grid}),
    )
}

fn main() -> Result<()> {
    training::rewrite_sweep(PRODUCER, data::REWRITE_PROTOCOL, intervention)
}

//! Ordinary full-vocabulary CE through the fixed shared complex head. Every
//! original record contributes with its actual batch denominator. No early
//! stopping, best-checkpoint replacement or restored training logits.
use crate::*;
use features::{Encoder, Row};
pub fn policies() -> Result<Vec<Policy>> {
    Ok(serde_json::from_value(
        serde_json::from_str::<serde_json::Value>(PROTOCOL)?["training"].clone(),
    )?)
}
pub fn gradient(model: &Model, rows: &[Row], batch: &[usize]) -> Result<(f64, Vec<f64>)> {
    if batch.is_empty() || batch.iter().any(|&i| i >= rows.len()) {
        return Err("invalid original-record batch".into());
    }
    let mut g = vec![0.; model.theta.len()];
    let mut residual = vec![0.; model.input.vocabulary];
    let mut loss = 0.;
    for &i in batch {
        let row = &rows[i];
        loss += ce_gradient(&model.logits_row(row), row.label, &mut residual);
        let labels = &model.decoder[row.block];
        if labels.len() == 2 {
            let contrast = (residual[labels[1]] - residual[labels[0]]) / batch.len() as f64;
            for (f, p) in row.phi.iter().enumerate() {
                g[2 * f] += p.re * contrast;
                g[2 * f + 1] -= p.im * contrast;
            }
            g[2 * model.bank.states() + row.block] += contrast / row.scale;
        }
    }
    if g.iter().any(|x| !x.is_finite()) || !loss.is_finite() {
        return Err("nonfinite ordinary CE gradient".into());
    }
    Ok((loss / batch.len() as f64, g))
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Step {
    pub epoch: usize,
    pub record_ids: Vec<usize>,
    pub batch_ce_before: f64,
    pub gradient_l2: f64,
    pub change_l2: f64,
    pub projection_lambda: f64,
    pub expanded_class_l1: f64,
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Work {
    pub gradient_batches: usize,
    pub gradient_records: usize,
    pub effective_updates: usize,
    pub projected_parameters: usize,
    pub ce_records: usize,
    pub independent_products: usize,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Outcome {
    pub input_sha256: String,
    pub model: Model,
    pub initial_model_sha256: String,
    pub initialization: fit::Report,
    pub policy: Policy,
    pub bounds: limits::Bounds,
    pub initial_ce: f64,
    pub final_ce: f64,
    pub epoch_ce: Vec<f64>,
    pub steps: Vec<Step>,
    pub work: Work,
    pub maximum_product_residual: f64,
}
pub fn audit(model: &Model, cell: usize) -> Result<f64> {
    let compact = model.logits_cell(cell)?;
    let expanded = model.independent_logits(cell)?;
    let residual = compact
        .iter()
        .zip(&expanded)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max);
    let scale = compact
        .iter()
        .chain(&expanded)
        .map(|x| x.abs())
        .fold(1., f64::max);
    if residual > 1e-10 * scale {
        return Err("compact and independently expanded complex products disagree".into());
    }
    Ok(residual)
}
pub fn run(input: Input, policy: Policy) -> Result<Outcome> {
    let bounds = limits::forecast(&input, &policy)?;
    let input_sha256 = digest(&input)?;
    let (mut model, initialization) = fit::run(&input)?;
    let initial_model_sha256 = digest(&model)?;
    let rows = Encoder::new(model.bank).records(&input);
    let all: Vec<_> = (0..rows.len()).collect();
    let initial_ce = fit::score(&model, &rows, &all).ce_sum / rows.len() as f64;
    let mut epoch_ce = vec![initial_ce];
    let mut steps = vec![];
    let mut work = Work {
        ce_records: rows.len(),
        ..Work::default()
    };
    for epoch in 0..policy.iterations {
        let width = if policy.mode == Mode::Full {
            rows.len()
        } else {
            policy.batch
        };
        for batch in all.chunks(width) {
            let (ce, g) = gradient(&model, &rows, batch)?;
            let before = model.theta.clone();
            for (x, g) in model.theta.iter_mut().zip(&g) {
                *x -= policy.learning_rate * g;
            }
            let projection_lambda = model.project()?;
            let expanded_class_l1 = model.validate()?;
            let change_l2 = model
                .theta
                .iter()
                .zip(&before)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            steps.push(Step {
                epoch,
                record_ids: batch.iter().map(|&i| input.records[i].id).collect(),
                batch_ce_before: ce,
                gradient_l2: g.iter().map(|g| g * g).sum::<f64>().sqrt(),
                change_l2,
                projection_lambda,
                expanded_class_l1,
            });
            work.gradient_batches += 1;
            work.gradient_records += batch.len();
            work.effective_updates += usize::from(change_l2 > 0.);
            work.projected_parameters += model.theta.len();
        }
        epoch_ce.push(fit::score(&model, &rows, &all).ce_sum / rows.len() as f64);
        work.ce_records += rows.len();
    }
    let mut maximum_product_residual = 0f64;
    for r in &input.records {
        maximum_product_residual = maximum_product_residual.max(audit(&model, r.cell)?);
        work.independent_products += 1;
    }
    if work.gradient_batches != bounds.gradient_batches {
        return Err("gradient work differs from forecast".into());
    }
    Ok(Outcome {
        input_sha256,
        model,
        initial_model_sha256,
        initialization,
        policy,
        bounds,
        initial_ce,
        final_ce: *epoch_ce.last().unwrap(),
        epoch_ce,
        steps,
        work,
        maximum_product_residual,
    })
}
pub fn replay(input: &Input, saved: &Outcome) -> Result<()> {
    if input != &saved.model.input || &run(input.clone(), saved.policy.clone())? != saved {
        return Err("shared geometry state, selection, history or work changed".into());
    }
    Ok(())
}

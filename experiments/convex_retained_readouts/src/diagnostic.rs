//! Read-only CM105 diagnostics on original training records after sealing.
//! Analytic diagnostics of forward-trained endpoints are not forward training.
use crate::{Result, cache::Cache, experiment, score};
use bilinear::{experiment as original, guard};
use joint::{
    capacity::{self, Table},
    diagnostic::direction,
};
use rotary::{atom::row, model::Model};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{fs, path::Path};

#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorBound {
    pub actual_gap: f64,
    pub estimated_gap: f64,
    pub coefficient_error_peak: f64,
    pub correction: f64,
    pub estimated_direction_actual_change: f64,
    pub actual_direction_upper: f64,
    pub strict_negative_condition: bool,
    pub gap_bound_excess: f64,
    pub direction_bound_excess: f64,
}
pub fn error_bound(model: &Model, a: &Table, h: &Table) -> Result<ErrorBound> {
    let (_, actual) = direction(model, a)?;
    let (target, estimate) = direction(model, h)?;
    let error = a
        .iter()
        .zip(h)
        .map(|(a, h)| {
            a.iter()
                .zip(h)
                .map(|(a, h)| (a[0] - h[0]).norm().max((a[1] - h[1]).norm()))
                .sum::<f64>()
        })
        .fold(0f64, f64::max);
    let correction = 2. * model.budget * error;
    let u = capacity::weighted(model);
    let v = capacity::weighted(&target);
    let change: f64 = u
        .iter()
        .zip(&v)
        .zip(a)
        .flat_map(|((u, v), a)| u.iter().zip(v).zip(a))
        .map(|((u, v), a)| row([v[0] - u[0], v[1] - u[1]], *a).re)
        .sum();
    let upper = -estimate.approximate_gap + correction;
    let gap_excess =
        ((actual.approximate_gap - estimate.approximate_gap).abs() - correction).max(0.);
    let direction_excess = (change - upper).max(0.);
    if !correction.is_finite()
        || !change.is_finite()
        || gap_excess > 1e-9
        || direction_excess > 1e-9
    {
        return Err("floating CM105 bound failed".into());
    }
    Ok(ErrorBound {
        actual_gap: actual.approximate_gap,
        estimated_gap: estimate.approximate_gap,
        coefficient_error_peak: error,
        correction,
        estimated_direction_actual_change: change,
        actual_direction_upper: upper,
        strict_negative_condition: estimate.approximate_gap > correction,
        gap_bound_excess: gap_excess,
        direction_bound_excess: direction_excess,
    })
}
pub fn run(workspace: &Path, storage: &Path, cohort: &Path, output: &Path) -> Result<()> {
    guard::admit(0, workspace, storage)?;
    score::sealed(cohort)?;
    fs::create_dir(output)?;
    for arm in experiment::ARMS {
        for variant in experiment::VARIANTS {
            for mode in experiment::MODES {
                let path = experiment::cell(cohort, arm, variant, mode);
                let (model, _, task) = score::load(workspace, storage, &path)?;
                let training = &task.splits["train"];
                let indices: Vec<_> = (0..training.len()).collect();
                let mut scan = joint::diagnostic::Scan::new(&model)?;
                for ids in indices.chunks(32) {
                    guard::admit(
                        model
                            .reservation(ids.len())?
                            .checked_mul(6)
                            .ok_or("diagnostic reservation")?,
                        workspace,
                        storage,
                    )?;
                    scan.observe(&original::records(training, ids)?)?;
                }
                let actual = scan.finish()?;
                let retained = experiment::settings(arm, variant, mode)?.2;
                let estimate = if retained {
                    let cache: Cache = serde_json::from_slice(&fs::read(path.join("cache.json"))?)?;
                    let records = original::records(training, &indices)?;
                    let recomputed = cache.recompute(&model, &records)?;
                    let drift = recomputed
                        .iter()
                        .flatten()
                        .flatten()
                        .zip(cache.mean.iter().flatten().flatten())
                        .map(|(a, b)| (*a - *b).norm())
                        .fold(0f64, f64::max);
                    if drift > 1e-10 {
                        return Err(
                            "incremental cache differs from retained-signal recomputation".into(),
                        );
                    }
                    let bound = error_bound(&model, &actual.coefficient, &cache.mean)?;
                    json!({"corrected_loss_lower_bound":actual.fixed.mean_ce-bound.estimated_gap-bound.correction,
                        "bound":bound,"cache_recomputation_error":drift,"coverage":cache.coverage(),"refreshes":cache.refreshed})
                } else {
                    serde_json::Value::Null
                };
                original::write_json(
                    &output.join(format!("{arm}-{variant}-{mode}.json")),
                    &json!({
                    "source_fingerprint":experiment::fingerprint(),"endpoint_sha256":original::hash(&fs::read(path.join("endpoint.json"))?),
                    "arm":arm,"variant":variant,"mode":mode,"records":actual.fixed.records,
                    "mean_ce":actual.fixed.mean_ce,"actual_joint_support":actual.support,"fixed_weight_gap":actual.fixed.approximate_gap,
                    "retained_estimate":estimate,"maximum_actual_logit_error":actual.maximum_actual_logit_error,
                    "diagnostic_ce_evaluations":4*actual.fixed.records,"updates_accepted":0,
                    "claim":"training-only floating diagnostics, not finite-rounding or unseen-answer certificates"}),
                )?;
            }
        }
    }
    Ok(())
}

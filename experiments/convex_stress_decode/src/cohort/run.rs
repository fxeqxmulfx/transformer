//! Evaluate a bounded proposal once, retaining imperfect and absent endpoints.
use super::*;
use std::time::Instant;
pub fn run(root: &Path, id: &str) -> Result<()> {
    let started = Instant::now();
    let c = case(id)?;
    let inv = inventory(root)?;
    let x = inputs(&c)?;
    execute(root, c, inv, x, started)
}
pub(super) fn execute(root: &Path, c: Case, inv: Value, x: Inputs, started: Instant) -> Result<()> {
    check_task(&inv, &c, &x)?;
    let out = root.join(c.id());
    fs::create_dir(&out)?;
    let mut r = metadata(&c, &inv, &x);
    write_new(&out.join("metadata.json"), &r)?;
    let mut proposal = construct(&x)?;
    let model = proposal.model.take();
    r["construction_seconds_including_data"] = started.elapsed().as_secs_f64().into();
    r["proposal_cost"] = serde_json::to_value(&proposal.cost)?;
    r["proposal_exact_fit"] = proposal.exact_fit.into();
    r["proposal_best_errors"] = serde_json::to_value(proposal.best_errors)?;
    r["stop_reason"] = proposal.reason.clone().into();
    write_new(&out.join("proposal.json"), &proposal)?;
    r["proposal_sha256"] = hash(&bytes(&out.join("proposal.json"), 8 * 1024 * 1024)?).into();
    if let Some(model) = model {
        let started_check = Instant::now();
        let audit = model.audit(&x.observations)?;
        r["spectral_and_train_check_seconds"] = started_check.elapsed().as_secs_f64().into();
        if audit["training_errors"] != r["proposal_best_errors"]
            || audit["exact_observed_fit"] != r["proposal_exact_fit"]
        {
            return Err("saved model audit differs from categorical search".into());
        }
        let started_metrics = Instant::now();
        let metrics = metrics_all(&model, &x.task)?;
        let ce = number(&metrics["train"], "loss")?;
        let excess = ce - ideal_loss(model.vocab);
        let error_fraction = number(&audit, "training_errors")? / x.observations.len() as f64;
        if (excess - error_fraction).abs() > 1e-10
            || (ce - number(&audit, "train_ce")?).abs() > 1e-10
        {
            return Err("original full-vocabulary CE differs from saved-table errors".into());
        }
        r["metrics_seconds"] = started_metrics.elapsed().as_secs_f64().into();
        r["status"] = "complete".into();
        r["audit"] = audit;
        r["metrics"] = metrics;
        r["ideal_ce_excess"] = excess.into();
        r["audit_examples_presented"] = x.observations.len().into();
        write_new(&out.join("model.json"), &model)?;
        r["model_sha256"] = hash(&bytes(
            &out.join("model.json"),
            32 * (model.n as u64).pow(2) + 1024 * 1024,
        )?)
        .into();
    } else {
        r["status"] = "proposal_ineligible".into();
    }
    r["total_seconds_including_data_and_proposal"] = started.elapsed().as_secs_f64().into();
    write_new(&out.join("result.json"), &r)
}

//! Reconstruct the saved endpoint from its splitting point, then audit the raw tables.
use super::*;
use crate::solver::{Costs, Solver};

pub fn replay(root: &Path, id: &str) -> Result<Value> {
    let c = case(id)?;
    let inv = inventory(root)?;
    let x = inputs(&c)?;
    replay_with(root, c, inv, x)
}
pub(super) fn replay_with(root: &Path, c: Case, inv: Value, x: Inputs) -> Result<Value> {
    check_task(&inv, &c, &x)?;
    let p = protocol()?;
    let s = schedule(&c, &x)?;
    let out = root.join(c.id());
    let result_bytes = bytes(&out.join("result.json"), 8 * 1024 * 1024)?;
    let r: Value = serde_json::from_slice(&result_bytes)?;
    let expected_metadata = metadata(&c, &inv, &x)?;
    if read(&out.join("metadata.json"))? != expected_metadata
        || expected_metadata
            .as_object()
            .ok_or("invalid metadata")?
            .iter()
            .any(|(key, value)| r[key] != *value)
    {
        return Err("changed paired provenance, data, decoder or schedule".into());
    }
    if r["status"] != "complete" {
        return Err("incomplete paired cell".into());
    }
    let model_bytes = bytes(
        &out.join("model.json"),
        table_limit(x.shape, x.shape.orbits + x.shape.vocab),
    )?;
    let state = bytes(
        &out.join("diagnostic_state.json"),
        table_limit(x.shape, x.shape.orbits),
    )?;
    if r["model_sha256"] != hash(&model_bytes) || r["diagnostic_state_sha256"] != hash(&state) {
        return Err("saved paired file digest mismatch".into());
    }
    let model: Model = serde_json::from_slice(&model_bytes)?;
    drop(model_bytes);
    model.validate_shape()?;
    let (radius, nuclear_budget) = budgets(&c.budget, x.shape.side)?;
    if model.shape != x.shape
        || model.decoder != x.decoder
        || (model.radius, model.nuclear_budget) != (radius, nuclear_budget)
    {
        return Err("saved paired model configuration mismatch".into());
    }
    let mode = Mode::parse(&c.arm)?;
    let mut solver = Solver::new(&model, x.observations.clone(), mode)?;
    solver.restore_diagnostic_state(serde_json::from_slice(&state)?)?;
    drop(state);
    let (snapshot, alpha, shadow_norms) = solver.snapshot()?;
    if snapshot != model {
        return Err("endpoint does not match the saved splitting point".into());
    }
    drop(snapshot);
    audit::close(&json!(alpha), &r["uniform_mixture_alpha"], 1e-12)?;
    audit::close(
        &serde_json::to_value(shadow_norms)?,
        &r["shadow_norms"],
        1e-8,
    )?;
    audit::close(
        &serde_json::to_value(model.validate(&solver.projector)?)?,
        &r["moment_norms"],
        1e-8,
    )?;
    audit::close(
        &json!(solver.full_residual()?),
        &r["full_operator_residual"],
        1e-9,
    )?;
    audit::close(&json!(solver.fit_defect(&model)), &r["fit_defect"], 1e-12)?;
    if metrics_all(&model, &x.task)? != r["metrics"] {
        return Err("saved paired metrics mismatch".into());
    }
    let updates = count(&r, "updates")?;
    let presented = match mode {
        Mode::Full => updates * s.count,
        Mode::Mini => {
            updates / s.per_epoch * s.count + ((updates % s.per_epoch) * s.batch).min(s.count)
        }
        Mode::Forward => s.count,
    };
    let gradients = if mode == Mode::Forward { 0 } else { presented };
    let costs: Costs = serde_json::from_value(r["costs"].clone())?;
    let excess = number(&r["metrics"]["train"], "loss")? - ideal_loss(x.shape.vocab);
    if updates == 0
        || updates > s.limit
        || (updates != s.limit && !updates.is_multiple_of(s.cadence))
        || r["examples_presented"] != presented
        || r["training_gradient_evaluations"] != gradients
        || (number(&r, "epochs")? - presented as f64 / s.count as f64).abs() > 1e-12
        || costs.gradient_evaluations != gradients as u64
        || costs.spectral_projections != updates as u64
        || costs.row_projections != 2 * updates as u64
        || costs.decompositions > (updates * (x.shape.orbits - 1)) as u64
        || number(&r, "step")? != solver.step
        || (number(&r, "ideal_ce_excess")? - excess).abs() > 1e-12
        || match r["stop_reason"].as_str() {
            Some("budget") => updates != s.limit,
            Some("ideal_ce_excess") => excess > number(&p, "ideal_ce_tolerance")?,
            _ => true,
        }
    {
        return Err("invalid paired stopping, exposure or cost record".into());
    }
    // Release the optimizer's buffers before the independent raw-coordinate audit.
    drop(solver);
    let audit = audit::validate(&model, &x.task, &r)?;
    Ok(
        json!({"replayed":true,"id":c.id(),"source_fingerprint":fingerprint(),
        "result_sha256":hash(&result_bytes),"status":r["status"],
        "model_sha256":r["model_sha256"],"diagnostic_state_sha256":r["diagnostic_state_sha256"],
        "audit":audit}),
    )
}

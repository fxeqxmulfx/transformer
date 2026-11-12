//! Repeat discovery and the whole update schedule, then audit saved coordinates.
use super::*;

pub fn replay(root: &Path, id: &str) -> Result<Value> {
    let c = case(id)?;
    replay_with(root, c.clone(), inventory(root)?, inputs(&c)?)
}
pub(super) fn replay_with(root: &Path, c: Case, inv: Value, x: Inputs) -> Result<Value> {
    check_task(&inv, &c, &x)?;
    let out = root.join(c.id());
    let result_bytes = bytes(&out.join("result.json"), 8 * 1024 * 1024)?;
    let r: Value = serde_json::from_slice(&result_bytes)?;
    let expected = metadata(&c, &inv, &x)?;
    if read(&out.join("metadata.json"))? != expected
        || expected
            .as_object()
            .ok_or("invalid metadata")?
            .iter()
            .any(|(key, value)| r[key] != *value)
    {
        return Err("changed certified provenance".into());
    }
    let found = discovery(&x)?;
    let saved_discovery = bytes(&out.join("discovery.json"), 64 * 1024 * 1024)?;
    if r["discovery_sha256"] != hash(&saved_discovery)
        || serde_json::from_slice::<Value>(&saved_discovery)? != serde_json::to_value(&found)?
        || r["discovery_costs"] != serde_json::to_value(found.costs)?
        || r["discovery_stop"] != serde_json::to_value(found.stop)?
    {
        return Err("changed training-only discovery or proposal ledger".into());
    }
    let mut seal = json!({"replayed":true,"id":c.id(),"source_fingerprint":fingerprint(),
        "result_sha256":hash(&result_bytes),"status":r["status"],"discovery_sha256":r["discovery_sha256"],
        "discovery_repeated":true});
    let Some(verified) = found.verified else {
        if r["status"] != "uncertified"
            || r["covered_cells"] != 0
            || ["model.json", "state.json", "trace.jsonl"]
                .iter()
                .any(|name| out.join(name).exists())
        {
            return Err("invalid refused certified cell".into());
        }
        return Ok(seal);
    };
    if r["status"] != "complete"
        || r["covered_cells"] != verified.covered_cells()
        || r["full_training_fit"] != verified.full_training_fit()
    {
        return Err("invalid certificate coverage record".into());
    }
    let shape = x.training.shape();
    let model_bytes = bytes(
        &out.join("model.json"),
        table_limit(shape, shape.orbits + shape.vocab),
    )?;
    let state_bytes = bytes(
        &out.join("state.json"),
        table_limit(shape, 2 * shape.orbits),
    )?;
    let trace_bytes = bytes(&out.join("trace.jsonl"), 64 * 1024 * 1024)?;
    if r["model_sha256"] != hash(&model_bytes)
        || r["state_sha256"] != hash(&state_bytes)
        || r["trace_sha256"] != hash(&trace_bytes)
    {
        return Err("certified file digest mismatch".into());
    }
    let model: Model = serde_json::from_slice(&model_bytes)?;
    let state: State = serde_json::from_slice(&state_bytes)?;
    let trace: Vec<Value> = std::str::from_utf8(&trace_bytes)?
        .lines()
        .map(serde_json::from_str)
        .collect::<std::result::Result<_, _>>()?;
    drop((model_bytes, state_bytes, trace_bytes));
    let p = protocol()?;
    let s = schedule(&c, &x)?;
    let mode = Mode::parse(&c.arm)?;
    let rounds = count(&p, "repair_rounds")?;
    let updates = count(&r, "updates")?;
    if updates == 0 || updates > s.limit {
        return Err("invalid update count".into());
    }
    let mut solver = Solver::new(&x.training, verified.certificate(), mode)?;
    let mut rng = Random::new(count(&p, "seed")? as u32 + 10_000);
    let mut order: Vec<_> = (0..s.count).collect();
    let (mut presented, mut snapshots, mut passes) = (0, 0, 0_u64);
    let mut last_endpoint = None;
    for update in 0..=updates {
        if update % s.cadence == 0 || update == s.limit {
            let e = solver.snapshot(rounds)?;
            passes += e.norm_evaluations;
            let train = metrics::evaluate(&e.model, &x.task.splits["train"])?;
            let excess = number(&train, "loss")? - ideal_loss(shape.vocab);
            let item = json!({"update":update,"train":train,"ideal_ce_excess":excess,
                "iterate_weight":e.iterate_weight,"norm_evaluations":e.norm_evaluations,
                "last_split_residual":solver.residual(),"costs":solver.state().costs});
            audit::close(
                &item,
                trace.get(snapshots).ok_or("missing snapshot")?,
                1e-12,
            )?;
            snapshots += 1;
            let stop =
                update > 0 && (excess <= number(&p, "ideal_ce_tolerance")? || update == s.limit);
            if stop != (update == updates) {
                return Err("changed training-only stop".into());
            }
            if update == updates {
                let reason = if excess <= number(&p, "ideal_ce_tolerance")? {
                    "ideal_ce_excess"
                } else {
                    "budget"
                };
                if r["stop_reason"] != reason {
                    return Err("changed stop reason".into());
                }
                last_endpoint = Some(e);
            }
        }
        if update < updates {
            let selected = batch(&solver, &x, &s, mode, update, &mut rng, &mut order);
            solver.step(&selected)?;
            if mode != Mode::Forward || update == 0 {
                presented += selected.len();
            }
        }
    }
    let endpoint = last_endpoint.ok_or("endpoint was not at a scheduled observation")?;
    let (residual, diagnostic) = solver.full_residual()?;
    if solver.state() != state
        || endpoint.model != model
        || snapshots != trace.len()
        || r["examples_presented"] != presented
        || r["costs"] != serde_json::to_value(state.costs)?
        || r["training_gradient_evaluations"] != state.costs.gradient_evaluations
        || r["snapshot_norm_evaluations"] != passes
        || r["setup_norm_evaluations"] != 1
        || r["setup_row_projections"] != 1
        || r["diagnostic_costs"] != serde_json::to_value(diagnostic)?
    {
        return Err("changed certified state, endpoint, trace or work ledger".into());
    }
    for (key, value) in [
        ("iterate_weight", endpoint.iterate_weight),
        ("fit_defect", fit_defect(&model, &x.observations)),
        ("full_operator_residual", residual),
        ("last_split_residual", solver.residual()),
        ("epochs", presented as f64 / s.count as f64),
    ] {
        audit::close(&json!(value), &r[key], 1e-12)?;
    }
    if metrics_all(&model, &x.task)? != r["metrics"] {
        return Err("changed split metrics".into());
    }
    audit::close(
        &serde_json::to_value(endpoint.norms)?,
        &r["moment_norms"],
        1e-9,
    )?;
    crate::endpoint::check_equalities(&model, &verified)?;
    drop(solver);
    seal["full_update_replay"] = true.into();
    seal["audit"] = audit::validate(&model, &x.task, &r)?;
    for key in ["model_sha256", "state_sha256", "trace_sha256"] {
        seal[key] = r[key].clone();
    }
    Ok(seal)
}

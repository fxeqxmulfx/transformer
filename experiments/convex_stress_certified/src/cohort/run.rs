//! Training-only discovery, constraints and stopping; held-out metrics afterwards.
use super::*;
use std::{io::Write, time::Instant};

pub fn run(root: &Path, id: &str) -> Result<()> {
    let started = Instant::now();
    let c = case(id)?;
    execute(root, c.clone(), inventory(root)?, inputs(&c)?, started)
}
pub(super) fn execute(root: &Path, c: Case, inv: Value, x: Inputs, started: Instant) -> Result<()> {
    check_task(&inv, &c, &x)?;
    let p = protocol()?;
    let s = schedule(&c, &x)?;
    let out = root.join(c.id());
    fs::create_dir(&out)?;
    let mut r = metadata(&c, &inv, &x)?;
    write_new(&out.join("metadata.json"), &r)?;
    let search_started = Instant::now();
    let found = discovery(&x)?;
    r["discovery_seconds"] = search_started.elapsed().as_secs_f64().into();
    write_new(&out.join("discovery.json"), &found)?;
    r["discovery_sha256"] = hash(&bytes(&out.join("discovery.json"), 64 * 1024 * 1024)?).into();
    r["discovery_costs"] = serde_json::to_value(found.costs)?;
    r["discovery_stop"] = serde_json::to_value(found.stop)?;
    let Some(verified) = found.verified else {
        r["status"] = "uncertified".into();
        r["covered_cells"] = 0.into();
        r["total_seconds_including_data"] = started.elapsed().as_secs_f64().into();
        return write_new(&out.join("result.json"), &r);
    };
    r["covered_cells"] = verified.covered_cells().into();
    r["full_training_fit"] = verified.full_training_fit().into();
    let mode = Mode::parse(&c.arm)?;
    let train_started = Instant::now();
    let mut solver = Solver::new(&x.training, verified.certificate(), mode)?;
    let rounds = count(&p, "repair_rounds")?;
    let mut trace = fs::File::create_new(out.join("trace.jsonl"))?;
    let mut order: Vec<_> = (0..s.count).collect();
    let mut rng = Random::new(count(&p, "seed")? as u32 + 10_000);
    let (mut updates, mut presented, mut norm_evaluations) = (0, 0, 0_u64);
    let (endpoint, stop) = loop {
        if updates % s.cadence == 0 || updates == s.limit {
            let endpoint = solver.snapshot(rounds)?;
            norm_evaluations += endpoint.norm_evaluations;
            let train = metrics::evaluate(&endpoint.model, &x.task.splits["train"])?;
            let excess = number(&train, "loss")? - ideal_loss(x.task.vocab);
            if excess < -1e-10 {
                return Err("CE below the proved lower bound".into());
            }
            writeln!(
                trace,
                "{}",
                json!({"update":updates,"train":train,"ideal_ce_excess":excess,
                "iterate_weight":endpoint.iterate_weight,"norm_evaluations":endpoint.norm_evaluations,
                "last_split_residual":solver.residual(),"costs":solver.state().costs})
            )?;
            trace.flush()?;
            println!(
                "update={updates} train_ce_excess={excess:.4e} iterate_weight={:.8}",
                endpoint.iterate_weight
            );
            if updates > 0 && excess <= number(&p, "ideal_ce_tolerance")? {
                break (endpoint, "ideal_ce_excess");
            }
            if updates == s.limit {
                break (endpoint, "budget");
            }
        }
        let selected = batch(&solver, &x, &s, mode, updates, &mut rng, &mut order);
        solver.step(&selected)?;
        if mode != Mode::Forward || updates == 0 {
            presented += selected.len();
        }
        updates += 1;
    };
    let state = solver.state();
    let (residual, diagnostic_costs) = solver.full_residual()?;
    r["status"] = "complete".into();
    r["updates"] = updates.into();
    r["stop_reason"] = stop.into();
    r["examples_presented"] = presented.into();
    r["epochs"] = (presented as f64 / s.count as f64).into();
    r["training_gradient_evaluations"] = state.costs.gradient_evaluations.into();
    r["costs"] = serde_json::to_value(state.costs)?;
    r["setup_norm_evaluations"] = 1.into();
    r["setup_row_projections"] = 1.into();
    r["snapshot_norm_evaluations"] = norm_evaluations.into();
    r["diagnostic_costs"] = serde_json::to_value(diagnostic_costs)?;
    r["iterate_weight"] = endpoint.iterate_weight.into();
    r["moment_norms"] = serde_json::to_value(&endpoint.norms)?;
    r["fit_defect"] = fit_defect(&endpoint.model, &x.observations).into();
    r["last_split_residual"] = solver.residual().into();
    r["full_operator_residual"] = residual.into();
    r["elapsed_seconds"] = train_started.elapsed().as_secs_f64().into();
    // The selected endpoint is fixed before accessing any held-out metric.
    r["metrics"] = metrics_all(&endpoint.model, &x.task)?;
    r["ideal_ce_excess"] =
        (number(&r["metrics"]["train"], "loss")? - ideal_loss(x.task.vocab)).into();
    write_new(&out.join("model.json"), &endpoint.model)?;
    write_new(&out.join("state.json"), &state)?;
    for name in ["model", "state"] {
        let shape = x.training.shape();
        let width = if name == "model" {
            shape.orbits + shape.vocab
        } else {
            2 * shape.orbits
        };
        r[format!("{name}_sha256")] = hash(&bytes(
            &out.join(format!("{name}.json")),
            table_limit(shape, width),
        )?)
        .into();
    }
    r["trace_sha256"] = hash(&bytes(&out.join("trace.jsonl"), 64 * 1024 * 1024)?).into();
    r["total_seconds_including_data"] = started.elapsed().as_secs_f64().into();
    write_new(&out.join("result.json"), &r)
}

//! Train one planned cell with the original batches, cadence and stopping rule.
//! A cell that never reaches the CE lower bound is saved with its cap and metrics.
use super::*;
use crate::solver::Solver;
use convex_stress_relabel::random::Random;
use std::{io::Write, time::Instant};

pub fn run(root: &Path, id: &str) -> Result<()> {
    let started = Instant::now();
    let c = case(id)?;
    let inv = inventory(root)?;
    let x = inputs(&c)?;
    execute(root, c, inv, x, started)
}
pub(super) fn execute(root: &Path, c: Case, inv: Value, x: Inputs, started: Instant) -> Result<()> {
    check_task(&inv, &c, &x)?;
    let p = protocol()?;
    let s = schedule(&c, &x)?;
    let tolerance = number(&p, "ideal_ce_tolerance")?;
    let mut r = metadata(&c, &inv, &x)?;
    let out = root.join(c.id());
    fs::create_dir(&out)?;
    write_new(&out.join("metadata.json"), &r)?;
    let data_seconds = started.elapsed().as_secs_f64();
    let (radius, nuclear_budget) = budgets(&c.budget, x.shape.side)?;
    let mode = Mode::parse(&c.arm)?;
    let initial = Model::uniform(x.shape, x.decoder.clone(), radius, nuclear_budget)?;
    let mut solver = Solver::new(&initial, x.observations.clone(), mode)?;
    drop(initial);
    let ideal = ideal_loss(x.shape.vocab);
    let mut trace = fs::File::create_new(out.join("trace.jsonl"))?;
    let mut order: Vec<_> = (0..s.count).collect();
    let mut rng = Random::new(count(&p, "seed")? as u32 + 10_000);
    let (mut updates, mut presented, mut minimum_alpha) = (0, 0, 1_f64);
    let mut first_primary_99 = Value::Null;
    let train_started = Instant::now();
    let (model, alpha, shadow_norms, stop_reason) = loop {
        if updates % s.cadence == 0 || updates == s.limit {
            let (model, alpha, norms) = solver.snapshot()?;
            minimum_alpha = minimum_alpha.min(alpha);
            let primary = crate::metrics::evaluate(&model, &x.task.splits[&x.task.selection])?;
            let train = crate::metrics::evaluate(&model, &x.task.splits["train"])?;
            let excess = number(&train, "loss")? - ideal;
            if excess < -1e-10 {
                return Err("CE below proved lower bound".into());
            }
            if count(&primary, "exact")? * 100 >= count(&primary, "examples")? * 99
                && first_primary_99.is_null()
            {
                first_primary_99 = updates.into();
            }
            writeln!(
                trace,
                "{}",
                json!({"update":updates,"seconds":train_started.elapsed().as_secs_f64(),
                "train":train,"primary":primary,"ideal_ce_excess":excess,
                "uniform_mixture_alpha":alpha,"last_split_residual":solver.residual,
                "costs":solver.costs})
            )?;
            trace.flush()?;
            println!("update={updates} train_ce_excess={excess:.4e} alpha={alpha:.8}");
            if updates > 0 && excess <= tolerance {
                break (model, alpha, norms, "ideal_ce_excess");
            }
            if updates == s.limit {
                break (model, alpha, norms, "budget");
            }
        }
        let within = updates % s.per_epoch;
        if mode == Mode::Mini && within == 0 {
            rng.shuffle(&mut order);
        }
        let selected = if mode == Mode::Mini {
            &order[within * s.batch..((within + 1) * s.batch).min(s.count)]
        } else {
            &order
        };
        solver.step(selected)?;
        if mode != Mode::Forward || updates == 0 {
            presented += selected.len();
        }
        updates += 1;
    };
    let norms = model.validate(&solver.projector)?;
    let gradients = if mode == Mode::Forward { 0 } else { presented };
    if solver.costs.gradient_evaluations != gradients as u64
        || solver.costs.spectral_projections != updates as u64
    {
        return Err("solver ledger differs from the schedule".into());
    }
    r["status"] = "complete".into();
    r["updates"] = updates.into();
    r["stop_reason"] = stop_reason.into();
    r["examples_presented"] = presented.into();
    r["training_gradient_evaluations"] = gradients.into();
    r["epochs"] = (presented as f64 / s.count as f64).into();
    r["costs"] = serde_json::to_value(solver.costs)?;
    r["step"] = solver.step.into();
    r["first_primary_99_update"] = first_primary_99;
    r["uniform_mixture_alpha"] = alpha.into();
    r["minimum_observed_alpha"] = minimum_alpha.into();
    r["shadow_norms"] = serde_json::to_value(shadow_norms)?;
    r["moment_norms"] = serde_json::to_value(norms)?;
    r["fit_defect"] = solver.fit_defect(&model).into();
    r["last_split_residual"] = solver.residual.into();
    r["full_operator_residual"] = solver.full_residual()?.into();
    r["metrics"] = metrics_all(&model, &x.task)?;
    r["ideal_ce_excess"] = (number(&r["metrics"]["train"], "loss")? - ideal).into();
    r["data_seconds"] = data_seconds.into();
    r["elapsed_seconds"] = train_started.elapsed().as_secs_f64().into();
    write_new(&out.join("model.json"), &model)?;
    write_new(
        &out.join("diagnostic_state.json"),
        &solver.diagnostic_state(),
    )?;
    let numbers = x.shape.orbits + x.shape.vocab;
    r["model_sha256"] = hash(&bytes(
        &out.join("model.json"),
        table_limit(x.shape, numbers),
    )?)
    .into();
    r["diagnostic_state_sha256"] = hash(&bytes(
        &out.join("diagnostic_state.json"),
        table_limit(x.shape, x.shape.orbits),
    )?)
    .into();
    r["total_seconds_including_data"] = started.elapsed().as_secs_f64().into();
    write_new(&out.join("result.json"), &r)
}

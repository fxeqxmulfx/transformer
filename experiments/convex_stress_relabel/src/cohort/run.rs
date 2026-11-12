//! Original CM5/CM9 optimizer, cadence and CE, in a fixed CM12 coding.
use super::*;
use crate::{Mode, Model, Solver, Table, budgets, ideal_loss, metrics, random::Random};
use std::{io::Write, time::Instant};

pub fn run(root: &Path, id: &str) -> Result<()> {
    let started = Instant::now();
    let c = case(id)?;
    let inv = inventory(root)?;
    let p = protocol()?;
    let x = inputs(&c)?;
    execute(root, c, inv, p, x, started)
}

pub(super) fn execute(
    root: &Path,
    c: Case,
    inv: Value,
    p: Value,
    x: Inputs,
    started: Instant,
) -> Result<()> {
    check_inventory_task(&inv, &c.task, &x)?;
    let id = c.id();
    let proposal_seconds = started.elapsed().as_secs_f64();
    let n = x.task.vocab;
    let count = x.task.splits["train"].len();
    let batch = x.row["case"]["batch"].as_u64().unwrap() as usize;
    let per_epoch = count.div_ceil(batch);
    let cadence = (x.row["case"]["observe"].as_u64().unwrap() as usize).min(per_epoch);
    let mut record = json!({"case":c,"id":id,"source_fingerprint":fingerprint(),"protocol":p,"producer_commit":inv["producer_commit"],
        "vocab":n,"train_prefixes":count,"data_fingerprints":x.row["data_fingerprints"],"selection":x.task.selection,"discovery":x.discovery,
        "coding":x.coding,"explicit_bytes_upper_bound":x.explicit_bytes,"proposal_seconds_including_data":proposal_seconds,"proposal_examples_presented":count,
        "seed":p["seed"],"batch_size":batch,"updates_per_epoch":per_epoch,"first_primary_99_update":null});
    let out = root.join(&id);
    fs::create_dir(&out)?;
    write_new(&out.join("metadata.json"), &record)?;
    let Some(coding) = x.coding else {
        record["status"] = "proposal_ineligible".into();
        record["training_gradient_evaluations"] = 0.into();
        record["updates"] = 0.into();
        record["examples_presented"] = 0.into();
        write_new(&out.join("result.json"), &record)?;
        return Ok(());
    };
    let (radius, nuclear_budget) = budgets(&c.budget, n)?;
    let initial = Table::new(n, radius, nuclear_budget)?;
    let mode = Mode::parse(&c.arm)?;
    let mut solver = Solver::new(
        &initial,
        x.samples,
        mode,
        p["frequency_workers"].as_u64().unwrap() as usize,
    )?;
    drop(initial);
    let mut trace = fs::File::create_new(out.join("trace.jsonl"))?;
    let mut order: Vec<_> = (0..count).collect();
    let mut rng = Random::new(p["seed"].as_u64().unwrap() as u32 + 10_000);
    let (mut updates, mut presented, mut minimum_alpha) = (0, 0, 1_f64);
    let train_started = Instant::now();
    let (model, alpha, shadow_norms, stop_reason) = loop {
        if updates % cadence == 0 || updates == c.limit(&p) {
            let (table, alpha, norms) = solver.snapshot()?;
            let model = Model {
                coding: coding.clone(),
                table,
            };
            minimum_alpha = minimum_alpha.min(alpha);
            let primary = metrics::evaluate(&model, &x.task.splits[&x.task.selection])?;
            let train = metrics::evaluate(&model, &x.task.splits["train"])?;
            let excess = number(&train, "loss")? - ideal_loss(n);
            if excess < -1e-10 {
                return Err("CE below proved lower bound".into());
            }
            if primary["exact"].as_u64().unwrap() * 100
                >= primary["examples"].as_u64().unwrap() * 99
                && record["first_primary_99_update"].is_null()
            {
                record["first_primary_99_update"] = updates.into();
            }
            writeln!(
                trace,
                "{}",
                json!({"update":updates,"seconds":train_started.elapsed().as_secs_f64(),"train":train,"primary":primary,"ideal_ce_excess":excess,"uniform_mixture_alpha":alpha,"last_split_residual":solver.residual})
            )?;
            trace.flush()?;
            println!("update={updates} train_ce_excess={excess:.4e} alpha={alpha:.8}");
            if updates > 0 && excess <= number(&p, "ideal_ce_tolerance")? {
                break (model, alpha, norms, "ideal_ce_excess");
            }
            if updates == c.limit(&p) {
                break (model, alpha, norms, "budget");
            }
        }
        let within = updates % per_epoch;
        if mode == Mode::Mini && within == 0 {
            rng.shuffle(&mut order);
        }
        let selected = if mode == Mode::Mini {
            &order[within * batch..((within + 1) * batch).min(count)]
        } else {
            &order
        };
        solver.step(selected)?;
        if mode != Mode::Forward || updates == 0 {
            presented += selected.len();
        }
        updates += 1;
    };
    let norms = model.table.validate(&mut solver.projector)?;
    record["status"] = "complete".into();
    record["radius"] = radius.into();
    record["nuclear_budget"] = nuclear_budget.into();
    record["updates"] = updates.into();
    record["stop_reason"] = stop_reason.into();
    record["examples_presented"] = presented.into();
    record["training_gradient_evaluations"] =
        if mode == Mode::Forward { 0 } else { presented }.into();
    record["epochs"] = (presented as f64 / count as f64).into();
    record["uniform_mixture_alpha"] = alpha.into();
    record["minimum_observed_alpha"] = minimum_alpha.into();
    record["shadow_norms"] = serde_json::to_value(shadow_norms)?;
    record["moment_norms"] = serde_json::to_value(norms)?;
    record["fit_defect"] = solver.fit_defect(&model.table).into();
    record["last_split_residual"] = solver.residual.into();
    record["full_operator_residual"] = solver.full_residual()?.into();
    record["metrics"] = metrics_all(&model, &x.task)?;
    record["ideal_ce_excess"] =
        (number(&record["metrics"]["train"], "loss")? - ideal_loss(n)).into();
    record["elapsed_seconds"] = train_started.elapsed().as_secs_f64().into();
    record["total_seconds_including_data_and_proposal"] = started.elapsed().as_secs_f64().into();
    write_new(&out.join("model.json"), &model)?;
    write_new(
        &out.join("diagnostic_state.json"),
        &solver.diagnostic_state(),
    )?;
    record["model_sha256"] = hash(&fs::read(out.join("model.json"))?).into();
    record["diagnostic_state_sha256"] = hash(&fs::read(out.join("diagnostic_state.json"))?).into();
    write_new(&out.join("result.json"), &record)
}

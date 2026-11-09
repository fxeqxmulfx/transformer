//! Original source trajectories; adaptation of class_geometry_wsd/main.rs.
//! No historical checkpoint/report is substituted. Inputs come from the exact
//! pinned initializer, split and sampler, and all new evidence has its own hash.
use crate::{
    Result,
    application::{Backend, Metrics},
    checkpoints, classes,
    domain::{
        data::{Data, Plan},
        model::{Arm, E, initialize},
    },
    dynamics,
    infrastructure::{engine::Engine, storage},
    inputs::require,
    measure, optimizer, probe,
};
use serde_json::{Value, json};
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
    time::Instant,
};

fn metric(m: Metrics) -> Value {
    json!({"loss":m.loss,"correct":m.correct,"count":m.count,"accuracy":m.correct as f64/m.count as f64})
}
pub fn source_preflight() -> Result<Value> {
    let data = Data::new(7);
    let plan = Plan::new(7, 10000);
    let initial = initialize(Arm::Plain, 7);
    let engine = Engine::new(Arm::Plain, &initial, &data, &plan)?;
    measure::validate(&engine, Arm::Plain, &initial, &data)
}
pub fn seed(seed: u64, path: &Path, started: Instant, charged: f64) -> Result<Value> {
    let data = Data::new(seed);
    let plan = Plan::new(seed, 10000);
    let initial = initialize(Arm::Plain, seed);
    let identity = json!({"seed":seed,"initial_sha256":storage::hash(&storage::floats(&initial)),
        "train_sha256":storage::hash(&storage::words(&data.train)),"validation_sha256":storage::hash(&storage::words(&data.validation)),
        "order_sha256":storage::hash(&storage::words(&plan.order))});
    storage::json_write(&path.join("identity.json"), &identity)?;
    let mut active = vec![false; 97];
    for row in data.train.as_chunks::<3>().0 {
        active[row[2] as usize] = true;
    }
    require(
        active.iter().filter(|&&v| v).count() == 33,
        "unexpected answer support",
    )?;
    let part = classes::Partition::new(active, 128);
    let mut engine = Engine::new(Arm::Plain, &initial, &data, &plan)?;
    let opt = optimizer::Optimizer::new()?;
    let independent = measure::validate(&engine, Arm::Plain, &initial, &data)?;
    let mut state = engine.state()?;
    let mut summary = dynamics::Summary::new(
        part.energies(&initial[..E].iter().map(|&v| v as f64).collect::<Vec<_>>()),
    );
    let mut trace = BufWriter::new(File::create_new(path.join("embeddings.bin"))?);
    for v in [1_u64, 10001, E as u64] {
        trace.write_all(&v.to_le_bytes())?;
    }
    trace.write_all(&storage::floats(&state.parameters[..E]))?;
    let mut updates = BufWriter::new(File::create_new(path.join("updates.jsonl"))?);
    let probes = checkpoints::probes();
    let endpoints = checkpoints::endpoints();
    let mut geometry: Vec<_> = probes
        .iter()
        .map(|s| json!({"step":s,"status":"pending"}))
        .collect();
    let mut saved: Vec<_> = endpoints
        .iter()
        .map(|s| json!({"step":s,"status":"pending"}))
        .collect();
    let (mut checks, mut metrics, mut matches) = (vec![], vec![], 0);
    for step in 0..=10000 {
        if charged + started.elapsed().as_secs_f64() >= 7200.0 {
            break;
        }
        if let Ok(i) = endpoints.binary_search(&step) {
            saved[i] = checkpoints::save(&path.join("checkpoints"), step, &state)?;
        }
        if step % 100 == 0 {
            metrics.push(
                json!({"epoch":step as f64/10.0,"step":step,"train":metric(engine.evaluate(false)?),
                "validation":metric(engine.evaluate(true)?)}),
            );
        }
        let at_probe = probes.iter().position(|&s| s == step);
        if let Some(i) = at_probe {
            let mut out = probe::run(
                &engine,
                &state,
                &data,
                &part,
                step,
                &path.join(format!("step-{step:06}")),
            )?;
            out["status"] = json!(if out["passed"] == true {
                "pass"
            } else {
                "fail"
            });
            geometry[i] = out;
            storage::json_write(
                &path.join("progress.json"),
                &json!({"seed":seed,"step":step,"metrics":metrics,
                "geometry":geometry,"checkpoints":saved,"dynamics":summary.json()}),
            )?;
            eprintln!(
                "seed={seed} step={step} geometry={} elapsed={:.1}s",
                geometry[i]["status"],
                charged + started.elapsed().as_secs_f64()
            );
        }
        if step == 10000 {
            break;
        }
        let at = plan.boundaries[step];
        let batch = plan.boundaries[step + 1] - at;
        engine.forward(false, at, batch, true)?;
        let (readout, full) = engine.audit_gradient(batch)?;
        let norm = opt.step(&engine, step)?;
        let mut next = engine.state()?;
        if at_probe.is_some() {
            checks.push(optimizer::check(
                &state,
                &next,
                &engine.gradients.read()?,
                norm,
                step,
            )?);
            let instrumented = storage::state_bytes(step + 1, &next);
            engine.restore(&state)?;
            engine.forward(false, at, batch, true)?;
            engine.backward(batch)?;
            let ordinary = opt.step(&engine, step)?;
            next = engine.state()?;
            require(
                instrumented == storage::state_bytes(step + 1, &next) && norm == ordinary,
                "instrumented update changed",
            )?;
            matches += 1;
        }
        let record = summary.record(&part, &state, &next, &readout, &full, step + 1, norm)?;
        serde_json::to_writer(&mut updates, &record)?;
        updates.write_all(b"\n")?;
        trace.write_all(&storage::floats(&next.parameters[..E]))?;
        state = next;
    }
    trace.flush()?;
    updates.flush()?;
    let complete = summary.updates == 10000;
    let passed = complete
        && summary.damping_failures == 0
        && geometry.iter().all(|v| v["status"] == "pass")
        && saved.iter().all(|v| v["status"] == "pass");
    Ok(
        json!({"seed":seed,"status":if passed {"pass"} else if !complete {"capped"} else {"fail"},
        "completed":summary.updates,"identity":identity,"partition":part.metadata(),"independent_source_checks":independent,
        "instrumented_matches":matches,"optimizer_checks":checks,"metrics":metrics,"geometry":geometry,"checkpoints":saved,
        "dynamics":summary.json(),"trace_sha256":crate::provenance::file_hash(&path.join("embeddings.bin"))?,
        "updates_sha256":crate::provenance::file_hash(&path.join("updates.jsonl"))?}),
    )
}

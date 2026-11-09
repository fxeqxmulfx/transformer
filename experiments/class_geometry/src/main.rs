//! Task #199: complete source trajectories and supervised class geometry.
include!(concat!(env!("OUT_DIR"), "/modules.rs"));
#[path = "../../residual_dynamics/src/algebra.rs"]
pub mod algebra;
#[path = "../../signed_transport/src/svd.rs"]
pub mod basis_svd;
mod classes;
mod dynamics;
#[path = "../../residual_dynamics/src/geometry.rs"]
pub mod geometry;
#[path = "../../residual_dynamics/src/gradient.rs"]
pub mod gradient;
mod groups;
#[path = "../../single_update/src/inputs.rs"]
pub mod inputs;
#[path = "../../single_update/src/measure.rs"]
pub mod measure;
#[path = "../../flat_approx/src/model.rs"]
pub mod model;
#[path = "../../flat_approx/fixtures/reference.rs"]
pub mod oracle;
mod probe;
mod reconstruction;
#[path = "../../residual_dynamics/src/svd.rs"]
pub mod svd;
use application::{Backend, Metrics};
use domain::{
    Result,
    data::{Data, Plan},
    model::{Arm, E, initialize},
};
use infrastructure::{engine::Engine, storage};
use inputs::require;
use serde_json::{Value, json};
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::Path,
    time::Instant,
};
const SEEDS: [u64; 8] = [7, 11, 1, 3, 5, 13, 17, 19];
fn steps() -> Vec<usize> {
    [0, 100, 250, 500, 750, 1000, 1500, 2000, 3000, 4000]
        .into_iter()
        .chain((5000..=10000).step_by(250))
        .collect()
}
fn metric(m: Metrics) -> Value {
    json!({"loss":m.loss,"correct":m.correct,"count":m.count,"accuracy":m.correct as f64/m.count as f64})
}
fn source_hash() -> String {
    storage::hash(
        &[
            include_bytes!("main.rs").as_slice(),
            include_bytes!("classes.rs"),
            include_bytes!("dynamics.rs"),
            include_bytes!("probe.rs"),
            include_bytes!("groups.rs"),
            include_bytes!("reconstruction.rs"),
            include_bytes!("instrumentation.rs"),
            include_bytes!("../Cargo.toml"),
            include_bytes!("../Cargo.lock"),
            include_bytes!("../PROTOCOL.md"),
            include_bytes!("../../flat_approx/src/model.rs"),
            include_bytes!("../../signed_transport/src/svd.rs"),
            inputs::source_hash().as_bytes(),
            env!("TRAINER_SOURCE_SHA256").as_bytes(),
        ]
        .concat(),
    )
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    require(
        args.len() == 3,
        "usage: rotate90-complex-kernite CHECKPOINT_ROOT TASK198_REPORT NEW_OUTPUT_DIRECTORY",
    )?;
    let root = Path::new(&args[0]).canonicalize()?;
    let output = Path::new(&args[2]);
    require(!output.exists(), "refusing to overwrite evidence")?;
    let mut ledger = inputs::Ledger::default();
    let (original, protocol) = inputs::protocol(&mut ledger, &root)?;
    let reference = ledger.json(&Path::new(&args[1]).canonicalize()?)?;
    require(reference["status"] == "pass", "task #198 reference failed")?;
    fs::create_dir(output)?;
    algebra::threads();
    let started = Instant::now();
    let mut report = json!({"status":"running","source_sha256":source_hash(),"protocol":include_str!("../PROTOCOL.md"),"original_protocol":original,"geometry_steps":steps(),"seeds":SEEDS.map(|seed|json!({"seed":seed,"status":"pending"})),"replay_hashes":[]});
    storage::json_write(&output.join("report.json"), &report)?;
    for (si, seed) in SEEDS.into_iter().enumerate() {
        if started.elapsed().as_secs_f64() >= 1800. {
            report["seeds"][si]["status"] = json!("capped");
            continue;
        }
        let path = output.join(format!("seed{seed}"));
        fs::create_dir(&path)?;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_seed(
                seed,
                &root,
                &path,
                &protocol,
                &reference,
                &mut ledger,
                &mut report,
                started,
            )
        }));
        report["seeds"][si] = match result {
            Ok(Ok(v)) => v,
            Ok(Err(e)) => json!({"seed":seed,"status":"error","error":e.to_string()}),
            Err(_) => json!({"seed":seed,"status":"error","error":"panic; see stderr"}),
        };
        storage::json_write(&path.join("result.json"), &report["seeds"][si])?;
        storage::json_write(&output.join("report.json"), &report)?;
    }
    report["inputs"] = ledger.verify()?;
    report["seconds"] = json!(started.elapsed().as_secs_f64());
    let failed = report["seeds"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|s| s["status"] != "pass")
        .count();
    report["failed"] = json!(failed);
    report["status"] = json!(if failed == 0 { "pass" } else { "fail" });
    storage::json_write(&output.join("report.json"), &report)?;
    require(failed == 0, "trajectory study failed; all slots retained")
}
#[allow(
    clippy::too_many_arguments,
    reason = "Explicit immutable provenance and one shared cohort clock"
)]
fn run_seed(
    seed: u64,
    root: &Path,
    path: &Path,
    protocol: &str,
    reference: &Value,
    ledger: &mut inputs::Ledger,
    report: &mut Value,
    started: Instant,
) -> Result<Value> {
    let data = Data::new(seed);
    let plan = Plan::new(seed, 10000);
    let probes = steps();
    let mut active = vec![false; 97];
    for r in data.train.as_chunks::<3>().0 {
        active[r[2] as usize] = true;
    }
    require(
        active.iter().filter(|&&x| x).count() == 33,
        "unexpected observed answer support",
    )?;
    let part = classes::Partition::new(active, 128);
    let initial = initialize(Arm::Plain, seed);
    let mut engine = Engine::new(Arm::Plain, &initial, &data, &plan)?;
    let mut saved = vec![];
    let mut history = None;
    if [7, 11].contains(&seed) {
        for step in [0, 5000, 10000] {
            saved.push(inputs::checkpoint(
                ledger,
                root,
                &inputs::Case {
                    arm: Arm::Plain,
                    seed,
                    step,
                    data: &data,
                    plan: &plan,
                    protocol,
                    forecast_output: path,
                },
            )?);
        }
        history = Some(ledger.json(&root.join(format!(
            "{}-seed{seed}/step-010000/history.json",
            Arm::Plain.name()
        )))?);
    }
    let independent = measure::validate(&engine, Arm::Plain, &initial, &data)?;
    let mut state = engine.state()?;
    let mut summary = dynamics::Summary::new(
        part.energies(&initial[..E].iter().map(|&x| x as f64).collect::<Vec<_>>()),
    );
    let mut trace = BufWriter::new(File::create_new(path.join("embeddings.bin"))?);
    for v in [1u64, 10001, E as u64] {
        trace.write_all(&v.to_le_bytes())?;
    }
    trace.write_all(&storage::floats(&state.parameters[..E]))?;
    let mut updates = BufWriter::new(File::create_new(path.join("updates.jsonl"))?);
    let mut metrics = vec![];
    let mut geometry: Vec<Value> = probes
        .iter()
        .map(|&step| json!({"step":step,"status":"pending"}))
        .collect();
    let mut matches = 0;
    let mut baseline_matches = 0;
    for step in 0..=10000 {
        if started.elapsed().as_secs_f64() >= 1800. {
            break;
        }
        if let Some(i) = [0, 5000, 10000].iter().position(|&s| s == step)
            && !saved.is_empty()
        {
            let hash = storage::hash(&storage::state_bytes(step, &state));
            require(
                saved[i].metadata["state_sha256"] == hash,
                "source checkpoint replay mismatch",
            )?;
            report["replay_hashes"]
                .as_array_mut()
                .unwrap()
                .push(json!({"seed":seed,"step":step,"sha256":hash,"matched":true}));
        }
        if step % 100 == 0 {
            let row = json!({"epoch":step as f64/10.,"step":step,"train":metric(engine.evaluate(false)?),"validation":metric(engine.evaluate(true)?)});
            if let Some(h) = &history {
                require(row == h[step / 100], "source history changed")?;
            }
            metrics.push(row);
        }
        let probe_index = probes.iter().position(|&s| s == step);
        if let Some(i) = probe_index {
            let mut out = probe::run(
                &engine,
                &state,
                &data,
                &part,
                step,
                &path.join(format!("step-{step:06}")),
            )?;
            if let Some(old) = reference["cases"]
                .as_array()
                .unwrap()
                .iter()
                .find(|c| c["seed"] == seed && c["step"] == step)
            {
                for family in ["hidden", "readout"] {
                    for split in ["train", "validation"] {
                        let current = out["coverage"][split]
                            .as_array()
                            .unwrap()
                            .iter()
                            .find(|c| c["family"] == family && c["rank"] == 32)
                            .unwrap();
                        let prior = old["audit"]["bridges"][family][split]
                            .as_array()
                            .unwrap()
                            .iter()
                            .find(|c| c["rank"] == 32 && c["method"] == "old_geometry")
                            .unwrap();
                        require(
                            current["certified"] == prior["certified"],
                            "original rank-32 geometry changed",
                        )?;
                    }
                }
                baseline_matches += 1;
            }
            out["status"] = json!(if out["passed"] == true {
                "pass"
            } else {
                "fail"
            });
            geometry[i] = out;
            storage::json_write(
                &path.join("progress.json"),
                &json!({"seed":seed,"step":step,"metrics":metrics,"geometry":geometry,"dynamics":summary.json()}),
            )?;
            eprintln!(
                "seed={seed} step={step} geometry={} elapsed={:.1}s",
                geometry[i]["status"],
                started.elapsed().as_secs_f64()
            );
        }
        if step == 10000 {
            break;
        }
        let at = plan.boundaries[step];
        let batch = plan.boundaries[step + 1] - at;
        engine.forward(false, at, batch, true)?;
        let (readout, full) = engine.audit_gradient(batch)?;
        engine.update(step + 1)?;
        let mut next = engine.state()?;
        if probe_index.is_some() {
            let instrumented = storage::state_bytes(step + 1, &next);
            engine.restore(&state)?;
            engine.train_step(at, batch, step + 1)?;
            next = engine.state()?;
            require(
                instrumented == storage::state_bytes(step + 1, &next),
                "instrumented source update changed",
            )?;
            matches += 1;
        }
        let record = summary.record(&part, &state, &next, &readout, &full, step + 1)?;
        serde_json::to_writer(&mut updates, &record)?;
        updates.write_all(b"\n")?;
        trace.write_all(&storage::floats(&next.parameters[..E]))?;
        state = next;
    }
    trace.flush()?;
    updates.flush()?;
    let completed = summary.updates;
    let passed = completed == 10000
        && summary.damping_failures == 0
        && geometry.iter().all(|v| v["status"] == "pass");
    let mut digest = sha2::Sha256::new();
    use sha2::Digest;
    let mut file = File::open(path.join("embeddings.bin"))?;
    let mut buffer = vec![0u8; 1024 * 1024];
    use std::io::Read;
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        digest.update(&buffer[..n]);
    }
    let trace_hash = digest
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    Ok(
        json!({"seed":seed,"status":if passed{"pass"}else if completed<10000{"capped"}else{"fail"},"completed":completed,"partition":part.metadata(),"initial_sha256":storage::hash(&storage::floats(&initial)),"train_sha256":storage::hash(&storage::words(&data.train)),"validation_sha256":storage::hash(&storage::words(&data.validation)),"order_sha256":storage::hash(&storage::words(&plan.order)),"independent_source_checks":independent,"instrumented_matches":matches,"task198_geometry_matches":baseline_matches,"trace_sha256":trace_hash,"updates_sha256":storage::hash(&fs::read(path.join("updates.jsonl"))?),"metrics":metrics,"geometry":geometry,"dynamics":summary.json()}),
    )
}

//! Task #200: same trajectories with the selected GPT main-model AdamW/WSD.
include!(concat!(env!("OUT_DIR"), "/modules.rs"));
#[path = "../../residual_dynamics/src/algebra.rs"]
pub mod algebra;
#[path = "../../signed_transport/src/svd.rs"]
pub mod basis_svd;
#[path = "../../class_geometry/src/classes.rs"]
mod classes;
mod dynamics;
#[path = "../../residual_dynamics/src/geometry.rs"]
pub mod geometry;
#[path = "../../residual_dynamics/src/gradient.rs"]
pub mod gradient;
#[path = "../../class_geometry/src/groups.rs"]
mod groups;
#[path = "../../single_update/src/inputs.rs"]
pub mod inputs;
#[path = "../../single_update/src/measure.rs"]
pub mod measure;
#[path = "../../flat_approx/src/model.rs"]
pub mod model;
mod optimizer;
mod optimizer_kernels;
#[path = "../../flat_approx/fixtures/reference.rs"]
pub mod oracle;
mod pairing;
#[path = "../../class_geometry/src/probe.rs"]
mod probe;
#[path = "../../class_geometry/src/reconstruction.rs"]
mod reconstruction;
mod schedule;
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
#[cfg(not(any(feature = "lr003", feature = "lr01")))]
const STUDY_PROTOCOL: &str = include_str!("../PROTOCOL.md");
#[cfg(any(feature = "lr003", feature = "lr01"))]
const STUDY_PROTOCOL: &str = include_str!("../../class_geometry_lr/PROTOCOL.md");
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
            include_bytes!("../../class_geometry/src/classes.rs"),
            include_bytes!("dynamics.rs"),
            include_bytes!("../../class_geometry/src/probe.rs"),
            include_bytes!("../../class_geometry/src/groups.rs"),
            include_bytes!("../../class_geometry/src/reconstruction.rs"),
            include_bytes!("optimizer.rs"),
            include_bytes!("pairing.rs"),
            include_bytes!("optimizer_kernels.rs"),
            include_bytes!("schedule.rs"),
            &schedule::PEAK.to_le_bytes(),
            include_bytes!("instrumentation.rs"),
            include_bytes!("../Cargo.toml"),
            include_bytes!("../Cargo.lock"),
            STUDY_PROTOCOL.as_bytes(),
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
    if args == ["preflight"] {
        println!("{}", optimizer::preflight()?);
        return Ok(());
    }
    require(
        args.len() == 3 || args.len() == 4,
        "usage: rotate90-complex-kernite preflight | CHECKPOINT_ROOT PAIRED_REPORT NEW_OUTPUT_DIRECTORY [TASK200_FAILED_ATTEMPT]",
    )?;
    let root = Path::new(&args[0]).canonicalize()?;
    let output = Path::new(&args[2]);
    require(!output.exists(), "refusing to overwrite evidence")?;
    let mut ledger = inputs::Ledger::default();
    let (original, protocol) = inputs::protocol(&mut ledger, &root)?;
    let reference = ledger.json(&Path::new(&args[1]).canonicalize()?)?;
    require(reference["status"] == "pass", "paired baseline failed")?;
    let prior = if args.len() == 4 {
        require(
            schedule::PEAK == 0.001,
            "repair belongs only to the original main-model cohort",
        )?;
        let root = Path::new(&args[3]).canonicalize()?;
        let bytes = ledger.read(&root.join("report.json"))?;
        require(
            storage::hash(&bytes)
                == "6102a576aa2fb85013016728cf5cb2b53b9e2d949272585d58ddd6bebb9f6de4",
            "unexpected repair source",
        )?;
        let old: Value = serde_json::from_slice(&bytes)?;
        for seed in [17u64, 19] {
            let row = old["seeds"]
                .as_array()
                .unwrap()
                .iter()
                .find(|s| s["seed"] == seed)
                .unwrap();
            require(
                row["error"] == "paired initial geometry changed",
                "repair source is not an initialization-only failure",
            )?;
            let baseline = Path::new(&args[1]).parent().unwrap();
            for file in ["state.bin", "margins.bin", "queries.json"] {
                require(
                    ledger.read(&root.join(format!("seed{seed}/step-000000/{file}")))?
                        == ledger.read(&baseline.join(format!("seed{seed}/step-000000/{file}")))?,
                    "initialization repair is not byte-identical",
                )?;
            }
        }
        Some((root, old))
    } else {
        None
    };
    let previous_seconds = prior
        .as_ref()
        .map_or(0., |(_, r)| r["seconds"].as_f64().unwrap());
    let gpt = Path::new("/home/misha/python_projects/gpt/research/main_model");
    for (path, hash) in [
        (
            "model.json",
            "0af933c3b6ec3e322714cd9654a6ab70b0d6bca193454a8b8dd165635d605e08",
        ),
        (
            "phase_score_native/src/optimizer.rs",
            "ceb430d1d5eff338e8826266f20c01870678e7655f0a99f7c62fbeb84b74974b",
        ),
        (
            "phase_score_native/src/optimizer_kernels.rs",
            "a5acfb922036ca76a555793d83a225a045ff9911e61ee927b96a19e6e2143258",
        ),
        (
            "wsd_suite_native/src/schedule.rs",
            "5853b9f7255428ef8fa7d11b7e98c329dc0129e0990c47430a77eeaef5c22bd4",
        ),
    ] {
        require(
            storage::hash(&ledger.read(&gpt.join(path))?) == hash,
            "GPT main optimizer source changed",
        )?;
    }
    fs::create_dir(output)?;
    algebra::threads();
    let started = Instant::now();
    let mut report = json!({"status":"running","source_sha256":source_hash(),"peak_lr":schedule::PEAK,"optimizer_ptx_sha256":storage::hash(optimizer_kernels::kernels::PTX.as_bytes()),"protocol":STUDY_PROTOCOL,"original_protocol":original,"geometry_steps":steps(),"seeds":SEEDS.map(|seed|json!({"seed":seed,"status":"pending"})),"replay_hashes":[]});
    report["prior_attempt"] = prior.as_ref().map_or(Value::Null, |(root,r)|json!({"path":root,"source_sha256":r["source_sha256"],"seconds":previous_seconds,"reason":"JSON ratio comparison; failed initial states and bounds byte-identical"}));
    report["measurement_sources"] = json!({});
    if let Some((_, old)) = &prior {
        report["replay_hashes"] = old["replay_hashes"].clone();
    }
    storage::json_write(&output.join("report.json"), &report)?;
    for (si, seed) in SEEDS.into_iter().enumerate() {
        if previous_seconds + started.elapsed().as_secs_f64() >= 1800. {
            report["seeds"][si]["status"] = json!("capped");
            continue;
        }
        let path = output.join(format!("seed{seed}"));
        if let Some((old_root, old)) = &prior {
            let row = &old["seeds"][si];
            if row["status"] == "pass" {
                require(
                    row["seed"] == seed && row["completed"] == 10000,
                    "incomplete reuse source",
                )?;
                require(
                    ledger.json(&old_root.join(format!("seed{seed}/result.json")))? == *row,
                    "reused seed summary changed",
                )?;
                std::os::unix::fs::symlink(old_root.join(format!("seed{seed}")), &path)?;
                report["seeds"][si] = row.clone();
                report["measurement_sources"][seed.to_string()] = old["source_sha256"].clone();
                storage::json_write(&output.join("report.json"), &report)?;
                continue;
            }
        }
        report["measurement_sources"][seed.to_string()] = json!(source_hash());
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
                previous_seconds,
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
    report["seconds"] = json!(previous_seconds + started.elapsed().as_secs_f64());
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
    previous_seconds: f64,
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
    let paired = reference["seeds"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["seed"] == seed)
        .ok_or("missing paired baseline seed")?;
    for (key, bytes) in [
        ("initial_sha256", storage::floats(&initial)),
        ("train_sha256", storage::words(&data.train)),
        ("validation_sha256", storage::words(&data.validation)),
        ("order_sha256", storage::words(&plan.order)),
    ] {
        require(paired[key] == storage::hash(&bytes), "paired input changed")?;
    }
    let mut engine = Engine::new(Arm::Plain, &initial, &data, &plan)?;
    let opt = optimizer::Optimizer::new()?;
    let mut optimizer_checks = vec![];
    let mut saved = vec![];
    if [7, 11].contains(&seed) {
        {
            let step = 0;
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
        if previous_seconds + started.elapsed().as_secs_f64() >= 1800. {
            break;
        }
        if step == 0 && !saved.is_empty() {
            let hash = storage::hash(&storage::state_bytes(step, &state));
            require(
                saved[0].metadata["state_sha256"] == hash,
                "source checkpoint replay mismatch",
            )?;
            report["replay_hashes"]
                .as_array_mut()
                .unwrap()
                .push(json!({"seed":seed,"step":step,"sha256":hash,"matched":true}));
        }
        if step % 100 == 0 {
            let row = json!({"epoch":step as f64/10.,"step":step,"train":metric(engine.evaluate(false)?),"validation":metric(engine.evaluate(true)?)});
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
            if step == 0 {
                require(
                    out["state_sha256"] == paired["geometry"][0]["state_sha256"]
                        && pairing::coverage_counts_equal(
                            &out["coverage"],
                            &paired["geometry"][0]["coverage"],
                        ),
                    "paired initial geometry changed",
                )?;
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
                previous_seconds + started.elapsed().as_secs_f64()
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
        if probe_index.is_some() {
            optimizer_checks.push(optimizer::check(
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
            let ordinary_norm = opt.step(&engine, step)?;
            next = engine.state()?;
            require(
                instrumented == storage::state_bytes(step + 1, &next) && norm == ordinary_norm,
                "instrumented source update changed",
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
        json!({"seed":seed,"status":if passed{"pass"}else if completed<10000{"capped"}else{"fail"},"completed":completed,"partition":part.metadata(),"initial_sha256":storage::hash(&storage::floats(&initial)),"train_sha256":storage::hash(&storage::words(&data.train)),"validation_sha256":storage::hash(&storage::words(&data.validation)),"order_sha256":storage::hash(&storage::words(&plan.order)),"independent_source_checks":independent,"instrumented_matches":matches,"paired_initial_matches":baseline_matches,"optimizer_checks":optimizer_checks,"trace_sha256":trace_hash,"updates_sha256":storage::hash(&fs::read(path.join("updates.jsonl"))?),"metrics":metrics,"geometry":geometry,"dynamics":summary.json()}),
    )
}

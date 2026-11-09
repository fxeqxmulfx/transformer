//! Task #206: one unchanged original update per frozen checkpoint.
include!(concat!(env!("OUT_DIR"), "/modules.rs"));
#[path = "../residual_dynamics/src/algebra.rs"]
pub mod algebra;
mod analysis;
#[path = "../class_geometry/src/classes.rs"]
pub mod classes;
#[path = "../geometry_gap/evidence.rs"]
pub mod evidence;
#[path = "../residual_dynamics/src/gradient.rs"]
pub mod gradient;
#[path = "../single_update/src/inputs.rs"]
pub mod inputs;
#[path = "../class_geometry_wsd/src/optimizer.rs"]
pub mod optimizer;
#[path = "../class_geometry_wsd/src/optimizer_kernels.rs"]
mod optimizer_kernels;
#[path = "../class_geometry_wsd/src/schedule.rs"]
pub mod schedule;
use application::{Backend, State};
pub use domain::Result;
use domain::{
    data::{Data, Plan},
    model::{Arm, E, initialize},
};
use evidence::Ledger;
use infrastructure::{engine::Engine, storage};
use inputs::require;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
    path::Path,
    time::Instant,
};
const SEEDS: [u64; 8] = [7, 11, 1, 3, 5, 13, 17, 19];
const ANCHORS: [usize; 6] = [0, 1000, 4000, 6000, 8000, 9500];
fn cohort() -> &'static str {
    if cfg!(feature = "settle") {
        "settle"
    } else if cfg!(feature = "lr01") {
        "lr01"
    } else {
        "lr001"
    }
}
fn source_hash() -> String {
    storage::hash(
        &[
            include_bytes!("main.rs").as_slice(),
            include_bytes!("analysis.rs"),
            include_bytes!("PROTOCOL.md"),
            include_bytes!("../class_geometry_wsd/src/optimizer.rs"),
            include_bytes!("../class_geometry_wsd/src/optimizer_kernels.rs"),
            include_bytes!("../class_geometry_wsd/src/schedule.rs"),
            include_bytes!("../class_geometry/src/classes.rs"),
            include_bytes!("../geometry_gap/evidence.rs"),
            include_bytes!("../class_geometry_wsd/Cargo.toml"),
            include_bytes!("../class_geometry_wsd/Cargo.lock"),
        ]
        .concat(),
    )
}
fn doubles(path: &Path, x: &[f64]) -> Result<()> {
    fs::write(
        path,
        x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
    )?;
    Ok(())
}
fn load_doubles(path: &Path) -> Result<Vec<f64>> {
    let b = fs::read(path)?;
    require(b.len().is_multiple_of(8), "double array shape")?;
    let v: Vec<_> = b
        .as_chunks::<8>()
        .0
        .iter()
        .map(|b| f64::from_le_bytes(*b))
        .collect();
    require(v.iter().all(|x| x.is_finite()), "nonfinite evidence")?;
    Ok(v)
}
fn hash_file(path: &Path) -> Result<String> {
    let mut f = File::open(path)?;
    let mut h = Sha256::new();
    let mut b = vec![0; 1 << 20];
    loop {
        let n = f.read(&mut b)?;
        if n == 0 {
            break;
        }
        h.update(&b[..n]);
    }
    Ok(h.finalize().iter().map(|x| format!("{x:02x}")).collect())
}
fn trace_row(path: &Path, step: usize) -> Result<Vec<f32>> {
    let mut f = File::open(path)?;
    let mut header = [0; 24];
    f.read_exact(&mut header)?;
    require(
        header
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| u64::from_le_bytes(*b))
            .collect::<Vec<_>>()
            == [1, 10001, E as u64],
        "trace shape",
    )?;
    f.seek(SeekFrom::Start((24 + 4 * E * step) as u64))?;
    let mut b = vec![0; 4 * E];
    f.read_exact(&mut b)?;
    Ok(b.as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect())
}
#[expect(
    clippy::too_many_arguments,
    reason = "Independent saved state and executed minibatch inputs"
)]
fn assess(
    old: &State,
    new: &State,
    features: &[f64],
    readout: &[f64],
    full: &[f64],
    rows: &[u32],
    norm: f32,
    step: usize,
) -> Result<Value> {
    let mut active = vec![false; 97];
    for c in 0u32..97 {
        active[(c * c * c % 97) as usize] = true;
    }
    // Caller checks this same partition against the complete training split.
    let part = classes::Partition::new(active, 128);
    let get = |x: &[f32]| x[..E].iter().map(|&v| v as f64).collect::<Vec<_>>();
    let (p, m, pn, mn, vn) = (
        get(&old.parameters),
        get(&old.first),
        get(&new.parameters),
        get(&new.first),
        get(&new.second),
    );
    let labels: Vec<_> = rows
        .as_chunks::<3>()
        .0
        .iter()
        .map(|r| r[2] as usize)
        .collect();
    let b = analysis::ce(features, &labels, &p, &part)?;
    require(
        b.maximum_feature_energy <= 128. * (1. + 5e-6),
        "normalized feature bound",
    )?;
    let error = b
        .gradient
        .iter()
        .zip(readout)
        .map(|(a, b)| (a - b).abs() / (1. + a.abs().max(b.abs())))
        .fold(0., f64::max);
    require(error <= 1e-5, "independent partial CE gradient mismatch")?;
    let clip = 1. / norm.max(1.) as f64;
    let g: Vec<_> = b.gradient.iter().map(|x| clip * x).collect();
    let enc: Vec<_> = full
        .iter()
        .zip(readout)
        .map(|(f, r)| clip * (f - r))
        .collect();
    let eta = schedule::Schedule::new(10000)?.rate(step) as f64;
    let mut v = analysis::diagnose(
        &part,
        &analysis::Step {
            old_p: &p,
            old_m: &m,
            new_p: &pn,
            new_m: &mn,
            new_v: &vn,
            readout: &g,
            encoder: &enc,
            eta,
            step: step + 1,
            l: clip * b.q * b.mean_feature_energy,
        },
    )?;
    v["q"] = json!(b.q);
    v["mean_feature_energy"] = json!(b.mean_feature_energy);
    v["max_feature_energy"] = json!(b.maximum_feature_energy);
    v["clip"] = json!(clip);
    v["partial_gradient_error"] = json!(error);
    v["eta"] = json!(eta);
    v["loss"] = json!(b.loss);
    Ok(v)
}
fn verify(out: &Path) -> Result<()> {
    let start = Instant::now();
    let report: Value = serde_json::from_slice(&fs::read(out.join("report.json"))?)?;
    require(
        report["status"] == "pass"
            && report["cohort"] == cohort()
            && report["source_sha256"] == source_hash(),
        "incomplete/different cohort",
    )?;
    let root = Path::new(report["source"].as_str().ok_or("source path")?);
    for (p, h) in report["inputs"].as_object().ok_or("input ledger")? {
        require(
            hash_file(Path::new(p))? == h.as_str().unwrap(),
            "source input changed",
        )?;
    }
    for c in report["cases"].as_array().ok_or("case inventory")? {
        let id = c["seed"].as_u64().unwrap();
        let step = c["anchor"].as_u64().unwrap() as usize;
        let path = out.join(format!("seed{id}-anchor{step}"));
        for (name, h) in c["evidence"].as_object().ok_or("case hashes")? {
            require(
                hash_file(&path.join(name))? == h.as_str().unwrap(),
                "evidence changed",
            )?;
        }
        let old = storage::decode(
            &fs::read(root.join(format!("seed{id}/step-{step:06}/state.bin")))?,
            Arm::Plain.count(),
        )?
        .1;
        let new = storage::decode(&fs::read(path.join("next.bin"))?, Arm::Plain.count())?.1;
        let rows: Vec<u32> = serde_json::from_slice(&fs::read(path.join("batch.json"))?)?;
        let recomputed = assess(
            &old,
            &new,
            &load_doubles(&path.join("features.bin"))?,
            &load_doubles(&path.join("readout.bin"))?,
            &load_doubles(&path.join("full.bin"))?,
            &rows,
            c["norm"].as_f64().unwrap() as f32,
            step,
        )?;
        require(recomputed == c["diagnosis"], "recomputed diagnosis changed")?;
        require(
            new.parameters[..E]
                == trace_row(&root.join(format!("seed{id}/embeddings.bin")), step + 1)?,
            "next trace mismatch",
        )?;
    }
    let total = report["seconds"].as_f64().unwrap() + start.elapsed().as_secs_f64();
    require(
        report["cases"].as_array().unwrap().len() == 48 && total <= 600.,
        "inventory/time cap",
    )?;
    println!(
        "{}",
        json!({"passed":true,"cohort":cohort(),"cases":48,"seconds":start.elapsed().as_secs_f64(),
        "combined_seconds":total,"source_sha256":source_hash()})
    );
    Ok(())
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() == 2 && args[0] == "verify" {
        return verify(Path::new(&args[1]));
    }
    require(
        args.len() == 2,
        "usage: ce-lyapunov NEW_OUTPUT SOURCE_COHORT | verify OUTPUT",
    )?;
    require(!cfg!(feature = "lr003"), "undeclared cohort")?;
    let out = Path::new(&args[0]);
    require(!out.exists(), "refusing evidence overwrite")?;
    let root = Path::new(&args[1]).canonicalize()?;
    let mut ledger = Ledger::default();
    let source = ledger.json(&root.join("report.json"))?;
    require(
        source["status"] == "pass" && source["seeds"].as_array().is_some_and(|v| v.len() == 8),
        "incomplete source",
    )?;
    fs::create_dir(out)?;
    let started = Instant::now();
    let mut report = json!({"status":"running","cohort":cohort(),"source":root,"source_sha256":source_hash(),
        "protocol":include_str!("PROTOCOL.md"),"cases":SEEDS.iter().flat_map(|seed|ANCHORS.iter().map(move |anchor|
            json!({"seed":seed,"anchor":anchor,"status":"pending"}))).collect::<Vec<_>>()});
    let opt = optimizer::Optimizer::new()?;
    report["trainer_source_sha256"] = json!(env!("TRAINER_SOURCE_SHA256"));
    report["optimizer_ptx_sha256"] =
        json!(storage::hash(optimizer_kernels::kernels::PTX.as_bytes()));
    let mut slot = 0;
    for (si, id) in SEEDS.into_iter().enumerate() {
        let data = Data::new(id);
        let plan = Plan::new(id, 10000);
        let initial = initialize(Arm::Plain, id);
        let sr = &source["seeds"][si];
        require(
            sr["seed"] == id && sr["status"] == "pass" && sr["completed"] == 10000,
            "source seed changed",
        )?;
        for (k, b) in [
            ("initial_sha256", storage::floats(&initial)),
            ("train_sha256", storage::words(&data.train)),
            ("validation_sha256", storage::words(&data.validation)),
            ("order_sha256", storage::words(&plan.order)),
        ] {
            require(sr[k] == storage::hash(&b), "source data/order changed")?;
        }
        let mut support = [false; 97];
        for r in data.train.as_chunks::<3>().0 {
            support[r[2] as usize] = true;
        }
        require(
            support.iter().filter(|&&x| x).count() == 33
                && (0u32..97).all(|c| support[(c * c * c % 97) as usize]),
            "training support changed",
        )?;
        let trace = root.join(format!("seed{id}/embeddings.bin"));
        require(
            hash_file(&trace)? == sr["trace_sha256"].as_str().unwrap(),
            "trace hash changed",
        )?;
        ledger
            .0
            .insert(trace.clone(), sr["trace_sha256"].as_str().unwrap().into());
        let mut engine = Engine::new(Arm::Plain, &initial, &data, &plan)?;
        for step in ANCHORS {
            let result = (|| -> Result<Value> {
                require(started.elapsed().as_secs_f64() < 600., "cohort cap reached")?;
                let bytes =
                    ledger.read(&root.join(format!("seed{id}/step-{step:06}/state.bin")))?;
                let (stored, old) = storage::decode(&bytes, Arm::Plain.count())?;
                let probe = sr["geometry"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|g| g["step"] == step)
                    .ok_or("missing anchor")?;
                require(
                    stored == step && storage::hash(&bytes) == probe["state_sha256"],
                    "anchor hash changed",
                )?;
                require(
                    old.parameters[..E] == trace_row(&trace, step)?,
                    "anchor trace mismatch",
                )?;
                engine.restore(&old)?;
                let at = plan.boundaries[step];
                let batch = plan.boundaries[step + 1] - at;
                let rows: Vec<u32> = plan.order[at..at + batch]
                    .iter()
                    .flat_map(|&i| {
                        data.train[3 * i as usize..3 * i as usize + 3]
                            .iter()
                            .copied()
                    })
                    .collect();
                engine.forward(false, at, batch, true)?;
                let features = engine.audit_features(batch)?.1;
                let (readout, full) = engine.audit_gradient(batch)?;
                let raw = engine.gradients.read()?;
                let norm = opt.step(&engine, step)?;
                let new = engine.state()?;
                let checked = optimizer::check(&old, &new, &raw, norm, step)?;
                require(
                    new.parameters[..E] == trace_row(&trace, step + 1)?,
                    "original next update changed",
                )?;
                let diagnosis = assess(&old, &new, &features, &readout, &full, &rows, norm, step)?;
                let path = out.join(format!("seed{id}-anchor{step}"));
                fs::create_dir(&path)?;
                fs::write(path.join("next.bin"), storage::state_bytes(step + 1, &new))?;
                doubles(&path.join("features.bin"), &features)?;
                doubles(&path.join("readout.bin"), &readout)?;
                doubles(&path.join("full.bin"), &full)?;
                evidence::write(&path.join("batch.json"), &json!(rows))?;
                let mut hashes = serde_json::Map::new();
                for name in [
                    "next.bin",
                    "features.bin",
                    "readout.bin",
                    "full.bin",
                    "batch.json",
                ] {
                    hashes.insert(name.into(), json!(hash_file(&path.join(name))?));
                }
                Ok(
                    json!({"norm":norm,"batch":batch,"optimizer":checked,"diagnosis":diagnosis,"evidence":hashes}),
                )
            })();
            match result {
                Ok(v) => {
                    report["cases"][slot]["status"] = json!("pass");
                    for (k, v) in v.as_object().unwrap() {
                        report["cases"][slot][k] = v.clone();
                    }
                }
                Err(e) => {
                    report["cases"][slot]["status"] = json!("error");
                    report["cases"][slot]["error"] = json!(e.to_string());
                }
            }
            slot += 1;
            evidence::write(&out.join("report.json"), &report)?;
            eprintln!(
                "CE replay cohort={} seed={id} anchor={step} slot={slot}/48 elapsed={:.1}s",
                cohort(),
                started.elapsed().as_secs_f64()
            );
        }
    }
    for (p, h) in &ledger.0 {
        require(hash_file(p)? == *h, "input changed during replay")?;
    }
    let failed = report["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["status"] != "pass")
        .count();
    report["status"] = json!(if failed == 0 { "pass" } else { "fail" });
    report["failed"] = json!(failed);
    report["inputs"] = json!(ledger.0);
    report["seconds"] = json!(started.elapsed().as_secs_f64());
    evidence::write(&out.join("report.json"), &report)?;
    require(failed == 0, "failed cases retained")
}

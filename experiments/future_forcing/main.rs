//! Task #205 construction phase: anchor states only; auditing is a separate command.
#[path = "../residual_dynamics/src/algebra.rs"]
pub mod algebra;
mod audit;
#[path = "../class_geometry/src/classes.rs"]
pub mod classes;
#[path = "../geometry_gap/evidence.rs"]
pub mod evidence;
mod forecast;
#[path = "../residual_dynamics/src/gradient.rs"]
pub mod gradient;
use evidence::{Ledger, hash, require, write};
use forecast::{Anchor, Forecast};
use serde_json::{Value, json};
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::Path,
    time::Instant,
};
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub const SEEDS: [u64; 8] = [7, 11, 1, 3, 5, 13, 17, 19];
pub const ANCHORS: [usize; 6] = [0, 1000, 4000, 6000, 8000, 9500];
fn source_hash() -> String {
    hash(
        &[
            include_bytes!("main.rs").as_slice(),
            include_bytes!("forecast.rs"),
            include_bytes!("audit.rs"),
            include_bytes!("PROTOCOL.md"),
            include_bytes!("../class_geometry_wsd/Cargo.toml"),
            include_bytes!("../class_geometry_wsd/Cargo.lock"),
            include_bytes!("../class_geometry/src/classes.rs"),
            include_bytes!("../geometry_gap/evidence.rs"),
        ]
        .concat(),
    )
}
pub fn support(queries: &Value) -> Result<Vec<bool>> {
    let mut a = vec![false; 97];
    let mut n = 0;
    for q in queries.as_array().ok_or("missing training support")? {
        if q[2] == 0 {
            let t = q[1].as_u64().ok_or("missing training label")? as usize;
            require(t < 97, "invalid label")?;
            a[t] = true;
            n += 1;
        }
    }
    require(
        n == 4704 && a.iter().filter(|&&a| a).count() == 33,
        "training inventory changed",
    )?;
    Ok(a)
}
fn construct(
    state: &[u8],
    active: Vec<bool>,
    anchor: usize,
    cohort: &str,
    out: &Path,
) -> Result<Value> {
    let a = Anchor::new(forecast::state(state, anchor)?, active)?;
    let end = (anchor + 4000).min(10000);
    let mut f = Forecast::new(&a, anchor, cohort);
    fs::create_dir(out)?;
    let path = out.join("forecast.jsonl");
    let mut w = BufWriter::new(File::create_new(&path)?);
    for h in 1..=end - anchor {
        writeln!(w, "{}", f.step(&a, h))?;
    }
    w.flush()?;
    w.get_ref().sync_all()?;
    let v = json!({"status":"frozen","anchor":anchor,"end":end,"prefixes":end-anchor,
        "state_sha256":hash(state),"forecast_sha256":hash(&fs::read(&path)?),"initial":a.metadata(),
        "frozen_before_trace_reads":true,"tracks":forecast::NAMES,"thresholds":forecast::THRESHOLDS,
        "format":"JSONL [step,eta,common_decay,tracks]; track=[total_direction,active_block_direction,max_direction,rounding_cap,parameter_cap,active_radius,total_radius,share_lower]"});
    write(&out.join("frozen.json"), &v)?;
    Ok(v)
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() == 2 && args[0] == "verify" {
        return audit::all(Path::new(&args[1]));
    }
    require(
        args.len() == 4,
        "usage: future-forcing NEW_OUTPUT LR001 LR01 SETTLE",
    )?;
    let out = Path::new(&args[0]);
    require(!out.exists(), "refusing to overwrite evidence")?;
    let mut ledger = Ledger::default();
    let mut cohorts = vec![];
    for (i, name) in ["lr001", "lr01", "settle"].into_iter().enumerate() {
        let root = Path::new(&args[i + 1]).canonicalize()?;
        let r = ledger.json(&root.join("report.json"))?;
        require(
            r["status"] == "pass" && r["seeds"].as_array().is_some_and(|x| x.len() == 8),
            "incomplete source cohort",
        )?;
        cohorts.push((name, root, r));
    }
    fs::create_dir(out)?;
    let started = Instant::now();
    let mut slots = vec![];
    for (name, _, _) in &cohorts {
        for seed in SEEDS {
            for anchor in ANCHORS {
                slots.push(json!({"cohort":name,"seed":seed,"anchor":anchor,"end":(anchor+4000).min(10000),"status":"pending"}));
            }
        }
    }
    let mut report = json!({"status":"constructing","source_sha256":source_hash(),"protocol":include_str!("PROTOCOL.md"),
        "declared_cases":144,"declared_prefixes":444000,"cohorts":cohorts.iter().map(|(n,p,_)|json!({"name":n,"path":p})).collect::<Vec<_>>(),"cases":slots});
    let mut slot = 0;
    for (name, root, r) in &cohorts {
        for (si, seed) in SEEDS.into_iter().enumerate() {
            let reference = &r["seeds"][si];
            require(
                reference["seed"] == seed
                    && reference["completed"] == 10000
                    && reference["status"] == "pass",
                "seed inventory changed",
            )?;
            for key in [
                "initial_sha256",
                "train_sha256",
                "validation_sha256",
                "order_sha256",
            ] {
                require(
                    reference[key] == cohorts[0].2["seeds"][si][key],
                    "paired source differs",
                )?;
            }
            let source = root.join(format!("seed{seed}"));
            let active = support(&ledger.json(&source.join("step-000000/queries.json"))?)?;
            for anchor in ANCHORS {
                let result = (|| -> Result<Value> {
                    require(
                        started.elapsed().as_secs_f64() < 1800.,
                        "construction cap reached",
                    )?;
                    let path = source.join(format!("step-{anchor:06}/state.bin"));
                    let b = ledger.read(&path)?;
                    let probe = reference["geometry"]
                        .as_array()
                        .ok_or("missing probes")?
                        .iter()
                        .find(|v| v["step"] == anchor)
                        .ok_or("missing anchor")?;
                    require(hash(&b) == probe["state_sha256"], "anchor hash changed")?;
                    construct(
                        &b,
                        active.clone(),
                        anchor,
                        name,
                        &out.join(format!("{name}-seed{seed}-anchor{anchor}")),
                    )
                })();
                let (status, v) = match result {
                    Ok(v) => ("frozen", v),
                    Err(e) => ("error", json!({"error":e.to_string()})),
                };
                report["cases"][slot]["status"] = json!(status);
                report["cases"][slot]["construction"] = v;
                slot += 1;
            }
            eprintln!(
                "frozen future cohort={name} seed={seed} cases={slot}/144 elapsed={:.1}s",
                started.elapsed().as_secs_f64()
            );
            write(&out.join("report.json"), &report)?;
        }
    }
    ledger.verify()?;
    let failed = report["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["status"] != "frozen")
        .count();
    report["status"] = json!(if failed == 0 { "frozen" } else { "fail" });
    report["failed"] = json!(failed);
    report["construction_seconds"] = json!(started.elapsed().as_secs_f64());
    report["inputs"] = json!({"unchanged":true,"files":ledger.0});
    write(&out.join("report.json"), &report)?;
    require(failed == 0, "construction failed; every slot retained")
}

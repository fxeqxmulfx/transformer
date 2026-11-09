//! File reads are explicitly restricted to identity.json and the named anchor.
use crate::{
    Result,
    application::State,
    checkpoints,
    domain::{
        data::{Data, Plan},
        model::{Arm, initialize},
    },
    evidence::{self, Ledger, require},
    infrastructure::storage,
    prediction, provenance,
};
use serde_json::{Value, json};
use std::{fs, path::Path, time::Instant};
pub const COHORTS: [&str; 3] = ["lr001", "lr01", "settle"];
pub fn support(data: &Data) -> Vec<bool> {
    let mut active = vec![false; 97];
    for row in data.train.as_chunks::<3>().0 {
        active[row[2] as usize] = true;
    }
    active
}
pub fn old(
    root: &Path,
    seed: u64,
    anchor: usize,
    ledger: &mut Ledger,
) -> Result<(State, Data, Plan)> {
    let path = root.join(format!("seed{seed}"));
    let identity = ledger.json(&path.join("identity.json"))?;
    let data = Data::new(seed);
    let plan = Plan::new(seed, 10000);
    require(
        identity["seed"] == seed
            && identity["initial_sha256"]
                == storage::hash(&storage::floats(&initialize(Arm::Plain, seed)))
            && identity["train_sha256"] == storage::hash(&storage::words(&data.train))
            && identity["validation_sha256"] == storage::hash(&storage::words(&data.validation))
            && identity["order_sha256"] == storage::hash(&storage::words(&plan.order)),
        "original source identity",
    )?;
    require(
        support(&data).iter().filter(|v| **v).count() == 33,
        "answer support",
    )?;
    let bytes = ledger.read(&path.join(format!("checkpoints/step-{anchor:06}.bin")))?;
    let (step, state) = storage::decode(&bytes, Arm::Plain.count())?;
    require(step == anchor, "checkpoint time")?;
    Ok((state, data, plan))
}
pub fn run(out: &Path, roots: [&String; 3]) -> Result<()> {
    require(!out.exists(), "refusing forecast overwrite")?;
    fs::create_dir_all(out)?;
    let started = Instant::now();
    crate::algebra::threads();
    let cwd = std::env::current_dir()?;
    let roots: Vec<_> = roots.iter().map(|p| cwd.join(p)).collect();
    let cases:Vec<_>=COHORTS.into_iter().flat_map(|cohort|checkpoints::SEEDS.into_iter().flat_map(move|seed|
        checkpoints::ANCHORS.into_iter().map(move|anchor|json!({"cohort":cohort,"seed":seed,"anchor":anchor,"status":"pending",
            "horizons":checkpoints::HORIZONS.into_iter().filter(|h|anchor+h<=10000).collect::<Vec<_>>() })))).collect();
    let mut report = json!({"status":"running","source_sha256":provenance::source_hash(),"roots":roots,"cases":cases,
        "future_state_reads":0,"cap_seconds":1800});
    let report_path = out.join("construction.json");
    evidence::write(&report_path, &report)?;
    let mut ledger = Ledger::default();
    let result = (|| -> Result<()> {
        report["provenance"] = provenance::metadata()?;
        for (index, slot) in report["cases"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .enumerate()
        {
            let ci = index / 48;
            let seed = slot["seed"].as_u64().unwrap();
            let anchor = slot["anchor"].as_u64().unwrap() as usize;
            let result = (|| -> Result<Value> {
                require(started.elapsed().as_secs_f64() < 1800., "construction cap")?;
                let (state, data, plan) = old(&roots[ci], seed, anchor, &mut ledger)?;
                let payload = prediction::make(COHORTS[ci], seed, anchor, &state, &data, &plan)?;
                let name = format!("{}-seed{seed}-anchor{anchor}.json", COHORTS[ci]);
                let bytes = serde_json::to_vec(&payload)?;
                fs::write(out.join(&name), &bytes)?;
                Ok(json!({"file":name,"sha256":evidence::hash(&bytes)}))
            })();
            match result {
                Ok(value) => {
                    slot["status"] = json!("frozen");
                    slot["forecast"] = value;
                }
                Err(e) => {
                    slot["status"] = json!("error");
                    slot["error"] = json!(e.to_string());
                }
            }
            eprintln!(
                "constructed slot={index} status={} elapsed={:.1}s",
                slot["status"],
                started.elapsed().as_secs_f64()
            );
        }
        ledger.verify()?;
        Ok(())
    })();
    if let Err(e) = result {
        report["error"] = json!(e.to_string());
    }
    let failed = report["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["status"] != "frozen")
        .count();
    let seconds = started.elapsed().as_secs_f64();
    let pass = failed == 0 && report["error"].is_null() && seconds <= 1800.;
    report["status"] = json!(if pass { "frozen" } else { "fail" });
    report["failed"] = json!(failed);
    report["seconds"] = json!(seconds);
    report["inputs"] = json!(ledger.0);
    evidence::write(&report_path, &report)?;
    fs::write(
        out.join("construction.sha256"),
        evidence::hash(&fs::read(&report_path)?),
    )?;
    require(
        pass,
        "forecast construction failed; complete inventory retained",
    )
}

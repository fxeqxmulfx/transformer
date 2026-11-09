//! Only old-only source constructions and named anchor states are opened here.
use crate::{
    CAP, Result,
    evidence::{self, Ledger, require},
    prediction, provenance, sources,
};
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};
pub fn run(out: &Path, source_dirs: [&String; 2]) -> Result<()> {
    require(!out.exists(), "refusing forecast overwrite")?;
    fs::create_dir_all(out)?;
    let started = Instant::now();
    let dirs: [PathBuf; 2] = [
        Path::new(source_dirs[0]).canonicalize()?,
        Path::new(source_dirs[1]).canonicalize()?,
    ];
    let path = out.join("construction.json");
    let mut report = json!({"status":"running","source_sha256":provenance::source_hash(),"source_forecasts":dirs,
        "cases":sources::inventory(),"future_state_reads":0,"cap_seconds":CAP});
    evidence::write(&path, &report)?;
    let mut ledger = Ledger::default();
    let result = (|| -> Result<()> {
        report["provenance"] = provenance::metadata()?;
        let source = sources::pair(&dirs, &mut ledger)?;
        let roots: Vec<PathBuf> = serde_json::from_value(source[0]["roots"].clone())?;
        require(roots.len() == 3, "cohort roots")?;
        report["roots"] = json!(roots);
        for index in 0..144 {
            let spec = report["cases"][index].clone();
            let result = (|| -> Result<Value> {
                require(started.elapsed().as_secs_f64() < CAP, "construction budget")?;
                let seed = spec["seed"].as_u64().ok_or("seed")?;
                let anchor = spec["anchor"].as_u64().ok_or("anchor")? as usize;
                let (old, data, plan) =
                    sources::old(&roots[index / 48], seed, anchor, &mut ledger)?;
                let frozen = [
                    sources::payload(&dirs[0], &source[0]["cases"][index], &mut ledger)?,
                    sources::payload(&dirs[1], &source[1]["cases"][index], &mut ledger)?,
                ];
                let payload =
                    prediction::make(&old, &data, &plan, [&frozen[0], &frozen[1]], started, 0.)?;
                let name = format!(
                    "{}-seed{seed}-anchor{anchor}.json",
                    sources::COHORTS[index / 48]
                );
                let bytes = serde_json::to_vec(&payload)?;
                fs::write(out.join(&name), &bytes)?;
                Ok(json!({"file":name,"sha256":evidence::hash(&bytes)}))
            })();
            let slot = &mut report["cases"][index];
            match result {
                Ok(v) => {
                    slot["status"] = json!("frozen");
                    slot["forecast"] = v;
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
            evidence::write(&path, &report)?;
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
        .filter(|s| s["status"] != "frozen")
        .count();
    let seconds = started.elapsed().as_secs_f64();
    let pass = failed == 0 && report["error"].is_null() && seconds <= CAP;
    report["status"] = json!(if pass { "frozen" } else { "fail" });
    report["failed"] = json!(failed);
    report["seconds"] = json!(seconds);
    report["inputs"] = json!(ledger.0);
    evidence::write(&path, &report)?;
    fs::write(
        out.join("construction.sha256"),
        evidence::hash(&fs::read(&path)?),
    )?;
    require(pass, "construction failed; full inventory retained")
}

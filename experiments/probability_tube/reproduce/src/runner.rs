//! One cumulative cohort clock and a complete retained seed inventory.
use crate::{Result, checkpoints, infrastructure::storage, inputs::require, provenance, training};
use serde_json::{Value, json};
use std::{fs, path::Path, time::Instant};
pub fn run(root: &Path) -> Result<()> {
    fs::create_dir_all(root)?;
    let cohort = provenance::cohort()?;
    let mut prior = vec![];
    let mut charged = 60.0; // reserve for standalone source and optimizer preflight checks.
    let mut attempts: Vec<_> = fs::read_dir(root)?
        .map(|e| e.map(|e| e.path()))
        .collect::<std::io::Result<_>>()?;
    attempts.sort();
    for path in attempts {
        require(
            path.is_dir()
                && path
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .starts_with("attempt-"),
            "unexpected cohort-root entry",
        )?;
        let bytes = fs::read(path.join("report.json"))?;
        let report: Value = serde_json::from_slice(&bytes)?;
        require(
            report["cohort"] == cohort && report["status"] != "running",
            "incompatible or still-running prior attempt",
        )?;
        let seconds = report["attempt_seconds"]
            .as_f64()
            .ok_or("missing prior clock")?;
        require(seconds.is_finite() && seconds >= 0.0, "invalid prior clock")?;
        charged += seconds;
        prior.push(json!({"path":path,"report_sha256":storage::hash(&bytes),"seconds":seconds,"status":report["status"]}));
    }
    require(charged < 7200.0, "cohort time budget already exhausted")?;
    let path = root.join(format!("attempt-{:02}", prior.len() + 1));
    fs::create_dir(&path)?;
    let started = Instant::now();
    let inventory=checkpoints::SEEDS.map(|seed|json!({"seed":seed,"status":"pending",
        "geometry":checkpoints::probes().iter().map(|s|json!({"step":s,"status":"pending"})).collect::<Vec<_>>(),
        "checkpoints":checkpoints::endpoints().iter().map(|s|json!({"step":s,"status":"pending"})).collect::<Vec<_>>() }));
    let mut report = json!({"status":"running","cohort":cohort,"started_unix_seconds":provenance::unix_seconds(),
        "pid":std::process::id(),"prior_attempts":prior,"charged_seconds":charged,"cap_seconds":7200,
        "source_sha256":provenance::source_hash(),"seeds":inventory});
    let report_path = path.join("report.json");
    storage::json_write(&report_path, &report)?;
    let run = (|| -> Result<()> {
        report["provenance"] = provenance::metadata()?;
        crate::algebra::threads();
        report["optimizer_preflight"] = crate::optimizer::preflight()?;
        storage::json_write(&report_path, &report)?;
        for (i, seed) in checkpoints::SEEDS.into_iter().enumerate() {
            if charged + started.elapsed().as_secs_f64() >= 7200.0 {
                report["seeds"][i]["status"] = json!("capped");
                continue;
            }
            let directory = path.join(format!("seed{seed}"));
            fs::create_dir(&directory)?;
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                training::seed(seed, &directory, started, charged)
            }));
            match result {
                Ok(Ok(value)) => report["seeds"][i] = value,
                failed => {
                    report["seeds"][i]["status"] = json!("error");
                    report["seeds"][i]["error"] = json!(match failed {
                        Ok(Err(e)) => e.to_string(),
                        _ => "panic; see stderr".to_owned(),
                    });
                    if let Ok(bytes) = fs::read(directory.join("progress.json")) {
                        report["seeds"][i]["last_progress"] =
                            serde_json::from_slice::<Value>(&bytes)?;
                    }
                }
            }
            storage::json_write(&directory.join("result.json"), &report["seeds"][i])?;
            storage::json_write(&report_path, &report)?;
        }
        Ok(())
    })();
    if let Err(error) = run {
        report["error"] = json!(error.to_string());
    }
    let seconds = started.elapsed().as_secs_f64();
    let passed = report["error"].is_null()
        && report["seeds"]
            .as_array()
            .unwrap()
            .iter()
            .all(|s| s["status"] == "pass")
        && charged + seconds <= 7200.0;
    report["status"] = json!(if passed { "pass" } else { "fail" });
    report["attempt_seconds"] = json!(seconds);
    report["total_seconds"] = json!(charged + seconds);
    storage::json_write(&report_path, &report)?;
    require(
        passed,
        "reproduction failed or capped; all evidence retained",
    )
}

//! Preserve the producer report and explicitly mark a user-prioritized partial cohort.
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{error::Error, fs, path::Path};
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: record_interruption STOPPED_LR003_DIRECTORY UTC_TIME".into());
    }
    let root = Path::new(&args[0]);
    let bytes = fs::read(root.join("report.json"))?;
    let mut report: Value = serde_json::from_slice(&bytes)?;
    if report["status"] != "running" || report["peak_lr"] != 0.003 {
        return Err("not the stopped 0.003 cohort".into());
    }
    let archive = root.join("report-before-user-priority.json");
    if archive.exists() {
        return Err("interruption already recorded".into());
    }
    let mut completed = 0;
    let mut interrupted = 0;
    let mut unrun = 0;
    for row in report["seeds"].as_array_mut().ok_or("missing seeds")? {
        if row["status"] == "pass" {
            completed += 1;
            continue;
        }
        if row["status"] != "pending" {
            return Err("unexpected producer status".into());
        }
        let seed = row["seed"].as_u64().ok_or("missing seed")?;
        let progress = root.join(format!("seed{seed}/progress.json"));
        if progress.exists() {
            let saved: Value = serde_json::from_slice(&fs::read(progress)?)?;
            row["status"] = json!("interrupted_by_user_priority");
            row["last_saved_step"] = saved["step"].clone();
            interrupted += 1;
        } else {
            row["status"] = json!("unrun_after_user_priority");
            unrun += 1;
        }
    }
    if completed != 4 || interrupted != 1 || unrun != 3 {
        return Err("unexpected stopped slot inventory".into());
    }
    let digest: String = Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    report["status"] = json!("interrupted_by_user_priority");
    report["interruption"] = json!({"utc":args[1],"request":"0.01; start it","reason":"User prioritized immediate 0.01 training; preserve the stopped 0.003 cohort without treating missing seeds as measured outcomes","original_report_sha256":digest,"complete_seeds":completed,"interrupted_seeds":interrupted,"unrun_seeds":unrun});
    fs::write(archive, &bytes)?;
    // float_roundtrip preserves all measured numeric values while annotating status.
    fs::write(
        root.join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", report["interruption"]);
    Ok(())
}

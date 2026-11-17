//! Check every previous score against the archived decoders, verify the
//! lookup overlap and extend the original catalog with six greedy tasks.
#[path = "shared/audit.rs"]
mod archived;
#[path = "shared/overlap.rs"]
mod overlap;
use bilinear::guard;
use composed_retrieval::{Result, data};
use learned_retrieval::cohort::read;
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::Path, process::Command};
use synthetic_data::generative;

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let [work, storage, full, prefix, rewrite] = args.as_slice() else {
        return Err(
            "expected WORKSPACE STORAGE OFFSET_51_OUTPUT PREFIX_9_OUTPUT REWRITE_OUTPUT".into(),
        );
    };
    let [work, storage, full, prefix, rewrite] =
        [work, storage, full, prefix, rewrite].map(Path::new);
    guard::admit(1 << 30, work, storage)?;
    let previous = Command::new(prefix.join("original/target/release/reconcile-prefix"))
        .args([work, storage, full, prefix])
        .output()?;
    if !previous.status.success() {
        return Err(format!(
            "previous raw reconciliation failed: {}",
            String::from_utf8_lossy(&previous.stderr)
        )
        .into());
    }
    let mut ledger: Value = serde_json::from_slice(&previous.stdout)?;
    let p = data::protocol()?;
    for arm in ["original", "repeat"] {
        let seal = read(&rewrite.join(arm).join("run/sealed/manifest.json"))?;
        if seal["identity"]["protocol_sha256"] != baseline::hash(data::PROTOCOL.as_bytes()) {
            return Err("unexpected Rewrite training protocol".into());
        }
    }
    let audit = archived::audit(work, storage, rewrite)?;
    if ledger["summary"]["catalog"] != 165
        || ledger["summary"]["measured"] != 58
        || ledger["summary"]["source_unavailable"] != 6
        || audit["summary"]["measured"] != 7
        || audit["summary"]["unavailable_original_data"] != 0
    {
        return Err("unexpected prior or Rewrite coverage".into());
    }
    let cases = ledger["cases"]
        .as_object_mut()
        .ok_or("original case roster")?;
    if cases.len() != 165 {
        return Err("incomplete prior catalog".into());
    }
    let files = overlap::check_overlap(prefix, rewrite, "suite-lookup", &p)?;
    if files != 264 {
        return Err("incomplete shared lookup evidence".into());
    }
    let mut endpoints = BTreeMap::new();
    for (id, row) in audit["cases"].as_object().ok_or("Rewrite audit roster")? {
        let old = cases.get_mut(id).ok_or("unexpected original case")?;
        if old["control"] != row["catalog_control"]
            || p.cases.contains_key(id) == row["current"].is_null()
        {
            return Err("inconsistent original identity or source coverage".into());
        }
        if row["current"].is_null() {
            continue;
        }
        if id == "suite-lookup" {
            if old["current"] != row["current"] {
                return Err("lookup success differs".into());
            }
            continue;
        }
        if !generative::CASES.contains(&id.as_str()) || old["state"] != "outside_cohorts" {
            return Err("overlapping or unexpected Rewrite case".into());
        }
        old["current"] = row["current"].clone();
        old["state"] = json!("measured");
        let report = read(
            &rewrite
                .join("original/run")
                .join(id)
                .join("scores/report.json"),
        )?;
        let mut fits = BTreeMap::new();
        for mode in &p.modes {
            for seed in &p.seeds {
                let cell = format!("{mode}-{seed}-final");
                let mut splits = BTreeMap::new();
                for (name, split) in report["cells"][&cell]["splits"]
                    .as_object()
                    .ok_or("Rewrite splits")?
                {
                    let m = &split["metrics"];
                    let t = &m["teacher_forced"];
                    splits.insert(name.clone(),json!({"raw":[m["correct"],m["targets"]],"exact":[m["exact"],m["examples"]],
                        "teacher_raw":[t["correct"],t["targets"]],"teacher_exact":[t["exact"],t["examples"]],
                        "generated_tokens":m["generated_tokens"],"eos_rate":m["eos_rate"]}));
                }
                fits.insert(cell, splits);
            }
        }
        endpoints.insert(id, fits);
    }
    let count = |name: &str| {
        cases
            .values()
            .filter(|v| v["current"][name] == true)
            .count()
    };
    let summary = json!({"catalog":cases.len(),"measured":cases.values().filter(|x|x["state"]=="measured").count(),
        "source_unavailable":cases.values().filter(|x|x["state"]=="source_unavailable").count(),
        "outside_cohorts":cases.values().filter(|x|x["state"]=="outside_cohorts").count(),
        "original_pass_all":count("original"),"strict_pass_all":count("strict"),
        "rewrite_overlap":"suite-lookup","rewrite_overlap_raw_files":files});
    if endpoints.len() != 6 || summary["measured"] != 64 {
        return Err("incomplete six-case extension".into());
    }
    ledger["prefix_summary"] = ledger["summary"].clone();
    ledger["summary"] = summary;
    ledger["rewrite_audit"] = audit["summary"].clone();
    ledger["rewrite_endpoints"] = json!(endpoints);
    serde_json::to_writer_pretty(std::io::stdout().lock(), &ledger)?;
    println!();
    Ok(())
}

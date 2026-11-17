//! Reconcile the two independently audited CM116 cohorts, checking both
//! overlapping retrieval cases at every raw split before reporting coverage.
#[path = "shared/audit.rs"]
mod archived;
#[path = "shared/overlap.rs"]
mod overlap;
use bilinear::guard;
use composed_retrieval::{Result, data};
use learned_retrieval::cohort::read;
use serde_json::json;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let [work, storage, full, prefix] = args.as_slice() else {
        return Err("expected WORKSPACE STORAGE OFFSET_51_OUTPUT PREFIX_9_OUTPUT".into());
    };
    let [work, storage, full, prefix] = [work, storage, full, prefix].map(Path::new);
    guard::admit(1 << 30, work, storage)?;
    let p = data::protocol()?;
    for (root, def) in [(full, data::OFFSET_PROTOCOL), (prefix, data::PROTOCOL)] {
        for arm in ["original", "repeat"] {
            if read(&root.join(arm).join("run/sealed/manifest.json"))?["identity"]["protocol_sha256"]
                != baseline::hash(def.as_bytes())
            {
                return Err("unexpected training protocol".into());
            }
        }
    }
    let a = archived::audit(work, storage, full)?;
    let b = archived::audit(work, storage, prefix)?;
    if a["summary"]["measured"] != 51
        || b["summary"]["measured"] != 9
        || a["summary"]["unavailable_original_data"] != 6
        || b["summary"]["unavailable_original_data"] != 0
    {
        return Err("unexpected cohort or source availability".into());
    }
    let mut cases = BTreeMap::new();
    let mut shared = BTreeSet::new();
    let mut files = 0;
    let mut extra = BTreeMap::new();
    for (id, x) in a["cases"].as_object().ok_or("reference roster")? {
        let y = b["cases"].get(id).ok_or("missing original case")?;
        if x["catalog_control"] != y["catalog_control"]
            || x["historical_gpt_status"] != y["historical_gpt_status"]
        {
            return Err("historical case or control identity changed".into());
        }
        if !y["current"].is_null() {
            if !x["current"].is_null() {
                if x["current"] != y["current"] {
                    return Err("overlap success differs".into());
                }
                shared.insert(id.as_str());
                files += overlap::check_overlap(full, prefix, id, &p)?;
            }
            let report = read(
                &prefix
                    .join("original/run")
                    .join(id)
                    .join("scores/report.json"),
            )?;
            let mut endpoints = BTreeMap::new();
            for mode in &p.modes {
                for seed in &p.seeds {
                    let cell = format!("{mode}-{seed}-final");
                    let mut splits = BTreeMap::new();
                    for (name, row) in report["cells"][&cell]["splits"]
                        .as_object()
                        .ok_or("scored splits")?
                    {
                        let m = &row["metrics"];
                        splits.insert(name.clone(),json!({"raw":[m["correct"],m["targets"]],"exact":[m["exact"],m["examples"]]}));
                    }
                    endpoints.insert(cell, splits);
                }
            }
            extra.insert(id, endpoints);
        }
        let current = if y["current"].is_null() {
            &x["current"]
        } else {
            &y["current"]
        };
        let state = if !current.is_null() {
            "measured"
        } else if x["source_unavailable"] == true {
            "source_unavailable"
        } else {
            "outside_cohorts"
        };
        cases.insert(
            id,
            json!({"state":state,"current":current,"control":x["catalog_control"]}),
        );
    }
    if shared != BTreeSet::from(["suite-lookup", "suite-mqar"]) || files != 528 {
        return Err("incomplete overlap comparison".into());
    }
    let count = |f: &str| cases.values().filter(|v| v["current"][f] == true).count();
    let state = |s: &str| cases.values().filter(|v| v["state"] == s).count();
    let summary = json!({"catalog":cases.len(),"measured":state("measured"),"source_unavailable":state("source_unavailable"),
        "outside_cohorts":state("outside_cohorts"),"original_pass_all":count("original"),"strict_pass_all":count("strict"),
        "overlap":shared,"overlap_raw_files":files});
    serde_json::to_writer_pretty(
        std::io::stdout().lock(),
        &json!({"summary":summary,"cases":cases,"prefix_endpoints":extra,
        "baseline_audit":a["summary"],"prefix_audit":b["summary"]}),
    )?;
    println!();
    Ok(())
}

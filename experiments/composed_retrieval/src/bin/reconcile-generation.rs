//! Replay the 64-case and 24-case trained-radius studies from their own
//! frozen builds. Count the identical copy case once in the full catalog.
#[path = "shared/audit.rs"]
mod archived;
#[path = "shared/overlap.rs"]
mod overlap;
use baseline::hash;
use bilinear::guard;
use composed_retrieval::{Result, data, generation_cohort::OVERLAP};
use learned_retrieval::cohort::read;
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::Path};

fn outcomes(report: &Value, bound: Option<&Value>) -> Result<Value> {
    let mut fits = BTreeMap::new();
    for (cell, row) in report["cells"].as_object().ok_or("score endpoints")? {
        let mut splits = BTreeMap::new();
        for (name, split) in row["splits"].as_object().ok_or("scored splits")? {
            let m = &split["metrics"];
            let t = &m["teacher_forced"];
            if let Some(bounds) = bound {
                let b = &bounds[name];
                if m["examples"] != b[1]
                    || m["exact"].as_u64().ok_or("exact count")?
                        > b[0].as_u64().ok_or("control bound")?
                {
                    return Err("deterministic generation exceeds the source ceiling".into());
                }
            }
            splits.insert(name, json!({"raw":[m["correct"],m["targets"]],"exact":[m["exact"],m["examples"]],
                "teacher_raw":[t["correct"],t["targets"]],"teacher_exact":[t["exact"],t["examples"]],
                "final_answer_accuracy":m["final_answer_accuracy"],"generated_tokens":m["generated_tokens"],
                "extra_tokens":m["extra_tokens"],"eos_rate":m["eos_rate"]}));
        }
        if cell.ends_with("-final") {
            fits.insert(cell, splits);
        }
    }
    Ok(json!(fits))
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let [work, storage, prior, new] = args.as_slice() else {
        return Err("expected WORKSPACE STORAGE TRAINED_RADII_64 GENERATION_24".into());
    };
    let [work, storage, prior, new] = [work, storage, prior, new].map(Path::new);
    guard::admit(1 << 30, work, storage)?;
    for (directory, protocol) in [
        (prior, data::RADII_PROTOCOL),
        (new, data::GENERATION_PROTOCOL),
    ] {
        for arm in ["original", "repeat"] {
            let manifest = read(&directory.join(arm).join("run/sealed/manifest.json"))?;
            if manifest["identity"]["protocol_sha256"] != hash(protocol.as_bytes()) {
                return Err("unexpected frozen study".into());
            }
        }
    }
    let mut ledger = archived::audit(work, storage, prior)?;
    let audit = archived::audit(work, storage, new)?;
    if ledger["summary"]["measured"] != 64
        || ledger["summary"]["unavailable_original_data"] != 6
        || audit["summary"]["measured"] != 24
        || audit["summary"]["unavailable_original_data"] != 0
    {
        return Err("unexpected frozen source coverage".into());
    }
    let p = data::protocol()?;
    let raw = overlap::check_overlap(prior, new, OVERLAP, &p)?;
    if raw != 264 {
        return Err("incomplete shared copy evidence".into());
    }
    let inventory = read(&new.join("original/inventory.json"))?;
    let bounds = &inventory["random_lm_maximum_exact"];
    if bounds["validation/novel"] != json!([1, 64]) {
        return Err("missing source-derived control ceiling".into());
    }
    let cases = ledger["cases"].as_object_mut().ok_or("previous catalog")?;
    let mut endpoints = BTreeMap::new();
    for (id, row) in audit["cases"].as_object().ok_or("generation catalog")? {
        let old = cases.get_mut(id).ok_or("unknown case")?;
        if old["catalog_control"] != row["catalog_control"]
            || old["historical_gpt_status"] != row["historical_gpt_status"]
            || p.cases.contains_key(id) == row["current"].is_null()
        {
            return Err("inconsistent source identity".into());
        }
        if row["current"].is_null() {
            continue;
        }
        if id == OVERLAP {
            if old["current"] != row["current"] {
                return Err("copy criteria differ".into());
            }
            continue;
        }
        if !old["current"].is_null() || old["source_unavailable"] == true {
            return Err("unexpected overlapping or unavailable case".into());
        }
        let report = read(&new.join("original/run").join(id).join("scores/report.json"))?;
        endpoints.insert(
            id,
            outcomes(&report, (id == "random-lm-control").then_some(bounds))?,
        );
        old["current"] = row["current"].clone();
    }
    let measured = cases.values().filter(|r| !r["current"].is_null()).count();
    let unavailable = cases
        .values()
        .filter(|r| r["source_unavailable"] == true)
        .count();
    if measured != 87 || endpoints.len() != 23 || unavailable != 6 {
        return Err("incomplete combined coverage".into());
    }
    let summary = json!({"catalog":cases.len(),"measured":measured,"unavailable_original_data":unavailable,
        "outside_cohorts":cases.len()-measured-unavailable,"original_pass_all":cases.values().filter(|r|r["current"]["original"]==true).count(),
        "strict_pass_all":cases.values().filter(|r|r["current"]["strict"]==true).count(),
        "overlap":OVERLAP,"overlap_raw_files":raw});
    ledger["prior_summary"] = ledger["summary"].clone();
    ledger["generation_summary"] = audit["summary"].clone();
    ledger["summary"] = summary;
    ledger["random_lm_maximum_exact"] = bounds.clone();
    ledger["generation_endpoints"] = json!(endpoints);
    serde_json::to_writer_pretty(std::io::stdout().lock(), &ledger)?;
    println!();
    Ok(())
}

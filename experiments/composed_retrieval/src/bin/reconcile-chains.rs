//! Replay each independent study with its frozen raw decoder before adding
//! the final-answer chains to the same-model catalog.
#[path = "shared/audit.rs"]
mod archived;
#[path = "shared/overlap.rs"]
mod overlap;
use baseline::hash;
use bilinear::guard;
use composed_retrieval::{Result, chains, data};
use learned_retrieval::cohort::read;
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::Path, process::Command};

fn require_protocol(dir: &Path, protocol: &str) -> Result<()> {
    for arm in ["original", "repeat"] {
        let seal = read(&dir.join(arm).join("run/sealed/manifest.json"))?;
        if seal["identity"]["protocol_sha256"] != hash(protocol.as_bytes()) {
            return Err("unexpected frozen cohort protocol".into());
        }
    }
    Ok(())
}

fn counts(report: &Value, p: &data::Protocol) -> Result<Value> {
    let mut cells = BTreeMap::new();
    for mode in &p.modes {
        for seed in &p.seeds {
            let key = format!("{mode}-{seed}-final");
            let mut result = BTreeMap::new();
            for (name, n) in [
                ("train", 2048),
                ("validation", 1024),
                ("test/length-double", 1024),
            ] {
                let m = &report["cells"][&key]["splits"][name]["metrics"];
                if m["examples"] != n
                    || m["targets"] != n
                    || m["correct"] != m["exact"]
                    || m["exact"].as_u64().is_none_or(|v| v > n)
                {
                    return Err("final-answer exact and raw object counts disagree".into());
                }
                result.insert(name, json!([m["exact"], n]));
            }
            cells.insert(key, result);
        }
    }
    Ok(json!(cells))
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let paths: Vec<_> = args.iter().map(Path::new).collect();
    let [work, storage, parent, new, remaining @ ..] = paths.as_slice() else {
        return Err("expected WORKSPACE STORAGE TRAINED_RADII_64 CHAINS_13 [GENERATION_24]".into());
    };
    guard::admit(1 << 30, work, storage)?;
    require_protocol(parent, data::RADII_PROTOCOL)?;
    require_protocol(new, data::CHAIN_PROTOCOL)?;
    let (mut ledger, prior) = match remaining {
        [] => (archived::audit(work, storage, parent)?, 64),
        [gen_dir] => {
            require_protocol(gen_dir, data::GENERATION_PROTOCOL)?;
            let old = Command::new(gen_dir.join("original/target/release/reconcile-generation"))
                .args([*work, *storage, *parent, *gen_dir])
                .output()?;
            if !old.status.success() {
                return Err(format!(
                    "original generation reconciliation failed: {}",
                    String::from_utf8_lossy(&old.stderr)
                )
                .into());
            }
            (serde_json::from_slice(&old.stdout)?, 87)
        }
        _ => return Err("expected at most one original generation cohort".into()),
    };
    let audit = archived::audit(work, storage, new)?;
    if ledger["summary"]["measured"] != prior
        || ledger["summary"]["unavailable_original_data"] != 6
        || audit["summary"]["measured"] != 13
        || audit["summary"]["raw_files"] != 912
        || audit["summary"]["unavailable_original_data"] != 0
    {
        return Err("incomplete frozen studies".into());
    }
    let p = data::protocol()?;
    if p.name != chains::NAME {
        return Err("expected CM117 chain protocol".into());
    }
    let raw = overlap::check_overlap(parent, new, chains::OVERLAP, &p)?;
    if raw != 48 {
        return Err("incomplete shared addition raw comparison".into());
    }
    let mut endpoints = BTreeMap::new();
    let rows = ledger["cases"]
        .as_object_mut()
        .ok_or("original case catalog")?;
    for (id, r) in audit["cases"].as_object().ok_or("chain case catalog")? {
        let old = rows
            .get_mut(id)
            .ok_or("case absent from original catalog")?;
        if old["catalog_control"] != r["catalog_control"]
            || old["historical_gpt_status"] != r["historical_gpt_status"]
            || p.cases.contains_key(id) == r["current"].is_null()
        {
            return Err("original source identity differs".into());
        }
        if r["current"].is_null() {
            continue;
        }
        if id == chains::OVERLAP {
            if old["current"] != r["current"] {
                return Err("shared addition passes differ".into());
            }
            continue;
        }
        if !old["current"].is_null() || old["source_unavailable"] == true {
            return Err("unexpected overlapping or source-unavailable case".into());
        }
        let report = read(&new.join("original/run").join(id).join("scores/report.json"))?;
        endpoints.insert(id, counts(&report, &p)?);
        old["current"] = r["current"].clone();
    }
    let measured = rows.values().filter(|r| !r["current"].is_null()).count();
    let unavailable = rows
        .values()
        .filter(|r| r["source_unavailable"] == true)
        .count();
    if rows.len() != 165 || measured != prior + 12 || endpoints.len() != 12 || unavailable != 6 {
        return Err("incomplete reconciled chain ledger".into());
    }
    let summary = json!({"catalog":rows.len(),"measured":measured,"unavailable_original_data":unavailable,
      "outside_cohorts":rows.len()-measured-unavailable,"original_pass_all":rows.values().filter(|r|r["current"]["original"]==true).count(),
      "strict_pass_all":rows.values().filter(|r|r["current"]["strict"]==true).count(),"overlap":chains::OVERLAP,"overlap_raw_files":raw});
    ledger["prior_to_chains"] = ledger["summary"].clone();
    ledger["chain_summary"] = audit["summary"].clone();
    ledger["summary"] = summary;
    ledger["chain_exact_counts"] = json!(endpoints);
    serde_json::to_writer_pretty(std::io::stdout().lock(), &ledger)?;
    println!();
    Ok(())
}

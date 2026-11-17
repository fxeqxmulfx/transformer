//! Recount raw outputs in each frozen cohort and verify the overlapping
//! arithmetic-add run before combining coverage of the 165 original cases.
use bilinear::guard;
use composed_retrieval::{Result, data};
#[path = "shared/decisions.rs"]
mod decisions;
use decisions::Decisions;
use learned_retrieval::cohort::{raw_file, read};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
    process::Command,
};

fn audit(workspace: &Path, storage: &Path, root: &Path) -> Result<Value> {
    let out = Command::new(root.join("original/target/release/audit-raw"))
        .args([workspace, storage, root])
        .output()?;
    if !out.status.success() {
        return Err(format!("raw audit failed: {}", String::from_utf8_lossy(&out.stderr)).into());
    }
    let a: Value = serde_json::from_slice(&out.stdout)?;
    if a["producer"] != fs::read_to_string(root.join("producer.commit"))?.trim()
        || a["summary"]["catalog"] != 165
        || a["cases"].as_object().ok_or("case ledger")?.len() != 165
    {
        return Err("incomplete raw audit".into());
    }
    Ok(a)
}

fn check_overlap(a: &Path, b: &Path, id: &str, p: &data::Protocol) -> Result<usize> {
    let mut raw_files = 0;
    for arm in ["original", "repeat"] {
        let (a, b) = (
            a.join(arm).join("run").join(id),
            b.join(arm).join("run").join(id),
        );
        let (ra, rb) = (
            read(&a.join("scores/report.json"))?,
            read(&b.join("scores/report.json"))?,
        );
        if ra["cells"] != rb["cells"]
            || ra["metric"] != rb["metric"]
            || ra["target"] != rb["target"]
            || ra["selection"] != rb["selection"]
        {
            return Err("overlap original metrics or criteria differ".into());
        }
        for mode in &p.modes {
            for seed in &p.seeds {
                let cell = format!("{mode}-{seed}");
                let (x, y) = (
                    read(&a.join(&cell).join("receipt.json"))?,
                    read(&b.join(&cell).join("receipt.json"))?,
                );
                if x["data_fingerprints"] != y["data_fingerprints"]
                    || x["training"] != y["training"]
                    || fs::read(a.join(&cell).join("model.json"))?
                        != fs::read(b.join(&cell).join("model.json"))?
                {
                    return Err("overlap model, optimizer or source differs".into());
                }
                for phase in ["initial", "final"] {
                    for split in ["train", "validation"] {
                        let key = format!("{cell}-{phase}");
                        if fs::read(raw_file(&a.join("scores"), &key, split))?
                            != fs::read(raw_file(&b.join("scores"), &key, split))?
                        {
                            return Err("overlap raw logits differ".into());
                        }
                        raw_files += 1;
                    }
                }
            }
        }
    }
    Ok(raw_files)
}
fn elapsed(root: &Path, arm: &str, id: &str, cell: &str) -> Result<[f64; 4]> {
    let raw = fs::read_to_string(root.join(arm).join(format!("fit-{id}-{cell}.time")))?;
    let v = raw
        .split_whitespace()
        .map(str::parse::<f64>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let row: [f64; 4] = v
        .try_into()
        .map_err(|_| "expected elapsed, user CPU, system CPU and peak KiB")?;
    if row.iter().any(|x| !x.is_finite() || *x < 0.) {
        return Err("invalid resource receipt".into());
    }
    Ok(row)
}
fn family_cells(root: &Path, id: &str, p: &data::Protocol) -> Result<Value> {
    let dir = root.join("original/run").join(id);
    let scores = read(&dir.join("scores/report.json"))?;
    let mut out = BTreeMap::new();
    for mode in &p.modes {
        for seed in &p.seeds {
            let cell = format!("{mode}-{seed}");
            let receipt = read(&dir.join(&cell).join("receipt.json"))?;
            let e = &scores["cells"][format!("{cell}-final")];
            let counts = |split: &str| -> Result<[u64; 2]> {
                let m = &e["splits"][split]["metrics"];
                if m["correct"] != m["exact"] || m["targets"] != m["examples"] {
                    return Err("arithmetic mask or raw/exact match differs".into());
                }
                Ok([
                    m["correct"].as_u64().ok_or("correct")?,
                    m["targets"].as_u64().ok_or("targets")?,
                ])
            };
            let resources = ["original", "repeat"]
                .map(|arm| elapsed(root, arm, id, &cell))
                .into_iter()
                .collect::<Result<Vec<_>>>()?;
            out.insert(
                cell,
                json!({"train":counts("train")?,"validation":counts("validation")?,
            "ce_queries":receipt["training"]["forward_loss_queries"],"resources":resources}),
            );
        }
    }
    Ok(json!(out))
}
fn predictions(root: &Path, id: &str, p: &data::Protocol) -> Result<Decisions> {
    let scores = root.join("original/run").join(id).join("scores");
    let mut counts = Decisions::default();
    for seed in &p.seeds {
        for split in ["train", "validation"] {
            let raw =
                |mode: &str| fs::read(raw_file(&scores, &format!("{mode}-{seed}-final"), split));
            counts.add(&raw("mini")?, &raw("forward")?, p.cases[id].vocabulary)?;
        }
    }
    Ok(counts)
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let [workspace, storage, matched, family] = args.as_slice() else {
        return Err("expected WORKSPACE STORAGE MATCHED_FOUR_OUTPUT ARITHMETIC_OUTPUT".into());
    };
    let (workspace, storage, matched, family) = (
        Path::new(workspace),
        Path::new(storage),
        Path::new(matched),
        Path::new(family),
    );
    guard::admit(1 << 30, workspace, storage)?;
    let a = audit(workspace, storage, matched)?;
    let b = audit(workspace, storage, family)?;
    if a["summary"]["measured"] != 4
        || b["summary"]["measured"] != 48
        || b["summary"]["unavailable_original_data"] != 6
    {
        return Err("expected the frozen matched-four and arithmetic cohorts".into());
    }
    let p = data::protocol()?;
    let expected = [(matched, data::DENSE_PROTOCOL), (family, data::PROTOCOL)];
    for (root, definition) in expected {
        for arm in ["original", "repeat"] {
            let seal = read(&root.join(arm).join("run/sealed/manifest.json"))?;
            if seal["identity"]["protocol_sha256"] != baseline::hash(definition.as_bytes()) {
                return Err("unexpected training protocol".into());
            }
        }
    }
    let mut cases = BTreeMap::new();
    let mut family_results = BTreeMap::new();
    let mut family_predictions = BTreeMap::new();
    let mut total = Decisions::default();
    let mut overlap = BTreeSet::new();
    let mut raw_overlap_files = 0;
    for (id, v) in a["cases"].as_object().ok_or("four-case ledger")? {
        let w = b["cases"].get(id).ok_or("missing catalog case")?;
        if v["catalog_control"] != w["catalog_control"]
            || v["historical_gpt_status"] != w["historical_gpt_status"]
        {
            return Err("original catalog differs".into());
        }
        let (x, y) = (&v["current"], &w["current"]);
        if !y.is_null() {
            family_results.insert(id, family_cells(family, id, &p)?);
            let compared = predictions(family, id, &p)?;
            total.rows += compared.rows;
            total.different += compared.different;
            total.max_logit_delta = total.max_logit_delta.max(compared.max_logit_delta);
            family_predictions.insert(id, compared);
            if !x.is_null() {
                if x != y {
                    return Err("overlap outcome differs".into());
                }
                overlap.insert(id.as_str());
                raw_overlap_files += check_overlap(matched, family, id, &p)?;
            }
        }
        let current = if y.is_null() { x } else { y };
        let state = if !current.is_null() {
            "measured"
        } else if w["source_unavailable"] == true {
            "source_unavailable"
        } else {
            "outside_cohorts"
        };
        cases.insert(
            id,
            json!({"state":state,"control":v["catalog_control"],"current":current}),
        );
    }
    if overlap != BTreeSet::from(["arithmetic-add"]) {
        return Err("unexpected overlapping cases".into());
    }
    let count = |field: &str| {
        cases
            .values()
            .filter(|c| c["current"][field] == true)
            .count()
    };
    let state = |s: &str| cases.values().filter(|c| c["state"] == s).count();
    let summary = json!({"catalog":cases.len(),"measured":state("measured"),"source_unavailable":state("source_unavailable"),
        "outside_cohorts":state("outside_cohorts"),"original_pass_all":count("original"),"strict_pass_all":count("strict"),
        "overlap":overlap,"overlap_raw_files":raw_overlap_files,"arithmetic_mode_comparison":total});
    serde_json::to_writer_pretty(
        std::io::stdout().lock(),
        &json!({"summary":summary,"cases":cases,
        "four_case_audit":a["summary"],"arithmetic_audit":b["summary"],"arithmetic_endpoints":family_results,
        "arithmetic_mode_comparison":family_predictions}),
    )?;
    println!();
    Ok(())
}

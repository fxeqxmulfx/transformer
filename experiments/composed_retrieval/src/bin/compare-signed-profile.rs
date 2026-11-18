//! Decode each study with its own frozen binary. Compare matched initial
//! logits, source records, optimization budgets and all original split scores.
#[path = "shared/audit.rs"]
mod archived;
use baseline::hash;
use bilinear::guard;
use composed_retrieval::{Result, data, model::Model};
use learned_retrieval::cohort::{raw_file, read};
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, path::Path};

fn scores(r: &Value) -> Result<Value> {
    let mut counts = BTreeMap::new();
    for (k, v) in r["splits"].as_object().ok_or("score splits")? {
        let m = &v["metrics"];
        let t = m.get("teacher_forced").unwrap_or(m);
        counts.insert(
            k,
            json!({"raw":[m["correct"],m["targets"]],"exact":[m["exact"],m["examples"]],
            "teacher_raw":[t["correct"],t["targets"]],"ce":t["loss"]}),
        );
    }
    Ok(json!(counts))
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let [work, storage, old, new] = args.as_slice() else {
        return Err("expected WORKSPACE STORAGE RADII_64 SIGNED_PROFILE_10".into());
    };
    let [work, storage, old, new] = [work, storage, old, new].map(Path::new);
    guard::admit(1 << 30, work, storage)?;
    if data::PROTOCOL != data::PROFILE_PROTOCOL {
        return Err("expected signed profile build".into());
    }
    for (study, protocol) in [(old, data::RADII_PROTOCOL), (new, data::PROFILE_PROTOCOL)] {
        for arm in ["original", "repeat"] {
            let s = read(&study.join(arm).join("run/sealed/manifest.json"))?;
            if s["identity"]["protocol_sha256"] != hash(protocol.as_bytes()) {
                return Err("unexpected source study".into());
            }
        }
    }
    let base = archived::audit(work, storage, old)?;
    let mut ledger = archived::audit(work, storage, new)?;
    if base["summary"]["measured"] != 64
        || ledger["summary"]["measured"] != 10
        || ledger["summary"]["unavailable_original_data"] != 6
    {
        return Err("unexpected raw audit roster".into());
    }
    let p = data::protocol()?;
    let mut details = BTreeMap::new();
    let (mut initial_files, mut arithmetic_final_files) = (0, 0);
    for (id, s) in &p.cases {
        let (a, b) = (
            old.join("original/run").join(id),
            new.join("original/run").join(id),
        );
        let (ra, rb) = (
            read(&a.join("scores/report.json"))?,
            read(&b.join("scores/report.json"))?,
        );
        if ra["selection"] != rb["selection"]
            || ra["metric"] != rb["metric"]
            || ra["target"] != rb["target"]
            || base["cases"][id]["current"].is_null()
            || ledger["cases"][id]["current"].is_null()
        {
            return Err("changed original criterion or case".into());
        }
        let mut fits = BTreeMap::new();
        for mode in &p.modes {
            for seed in &p.seeds {
                let cell = format!("{mode}-{seed}");
                let init = format!("{cell}-initial");
                let trained = format!("{cell}-final");
                let (x, y) = (
                    read(&a.join(&cell).join("receipt.json"))?,
                    read(&b.join(&cell).join("receipt.json"))?,
                );
                let model: Model =
                    serde_json::from_slice(&fs::read(b.join(&cell).join("model.json"))?)?;
                let parent: Model =
                    serde_json::from_slice(&fs::read(a.join(&cell).join("model.json"))?)?;
                let l = model.validate()?;
                let lp = parent.validate()?;
                if !model.signed_profile
                    || parent.signed_profile
                    || !parent.learned_product_radii
                    || !model.learned_product_radii
                    || model.weights.len() - l.relation.len()
                        != parent.weights.len() - lp.relation.len()
                    || l.relation.len() != 2 * (s.context - 2).min(2)
                    || x["data_fingerprints"] != y["data_fingerprints"]
                    || ra["cells"][&init] != rb["cells"][&init]
                {
                    return Err(format!("changed initialization, sources or independent model dimensions on {id}/{cell}").into());
                }
                for k in [
                    "before",
                    "batch_sha256",
                    "examples",
                    "targets",
                    "updates",
                    "min_visits",
                    "max_visits",
                ] {
                    if x["training"][k] != y["training"][k] {
                        return Err(format!("training input or budget {k} differs").into());
                    }
                }
                for (m, r) in [(&model, &y), (&parent, &x)] {
                    let expected = if mode == "forward" {
                        2 * m.weights.len() * p.updates
                    } else {
                        0
                    };
                    if r["training"]["forward_loss_queries"] != expected {
                        return Err("forward query count differs".into());
                    }
                }
                for split in ra["cells"][&init]["splits"]
                    .as_object()
                    .ok_or("initial splits")?
                    .keys()
                {
                    let same = |key: &str| -> Result<bool> {
                        Ok(fs::read(raw_file(&a.join("scores"), key, split))?
                            == fs::read(raw_file(&b.join("scores"), key, split))?)
                    };
                    if !same(&init)? {
                        return Err("initial raw logits differ".into());
                    }
                    initial_files += 1;
                    if s.context == 2 {
                        if !same(&trained)? {
                            return Err("singleton positional control changed".into());
                        }
                        arithmetic_final_files += 1;
                    }
                }
                fits.insert(cell,json!({"coordinates":&model.weights[l.relation],"parent":scores(&ra["cells"][&trained])?,
                    "signed_profile":scores(&rb["cells"][&trained])?}));
            }
        }
        details.insert(id, json!({"parent":base["cases"][id]["current"],"signed_profile":ledger["cases"][id]["current"],"fits":fits}));
    }
    ledger["matched_parent"] = json!({"initial_raw_files":initial_files,"unchanged_arithmetic_final_files":arithmetic_final_files,
        "parent_summary":base["summary"],"cases":details});
    serde_json::to_writer_pretty(std::io::stdout().lock(), &ledger)?;
    println!();
    Ok(())
}

//! Compare the complete matched 64-case studies. Old duplicate cases are
//! reconciled by the archived decoder before selecting a reference directory.
#[path = "shared/audit.rs"]
mod archived;
use baseline::hash;
use bilinear::guard;
use composed_retrieval::{Result, data, model::Model, train};
use learned_retrieval::cohort::{raw_file, read};
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, path::Path, process::Command};

fn split_counts(entry: &Value) -> Result<Value> {
    let mut counts = BTreeMap::new();
    for (id, v) in entry["splits"]
        .as_object()
        .ok_or("held-out split metadata")?
    {
        let m = &v["metrics"];
        let mut row = json!({"raw":[m["correct"],m["targets"]],"exact":[m["exact"],m["examples"]]});
        if let Some(t) = m.get("teacher_forced") {
            row["teacher_raw"] = json!([t["correct"], t["targets"]]);
            row["generated_tokens"] = m["generated_tokens"].clone();
        }
        counts.insert(id, row);
    }
    Ok(json!(counts))
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let [work, storage, full, prefix, rewrite, changed] = args.as_slice() else {
        return Err(
            "expected WORKSPACE STORAGE OFFSET_51 PREFIX_9 REWRITE_7 TRAINED_RADII_64".into(),
        );
    };
    let [work, storage, full, prefix, rewrite, changed] =
        [work, storage, full, prefix, rewrite, changed].map(Path::new);
    guard::admit(1 << 30, work, storage)?;
    let old = Command::new(rewrite.join("original/target/release/reconcile-rewrite"))
        .args([work, storage, full, prefix, rewrite])
        .output()?;
    if !old.status.success() {
        return Err(format!(
            "prior raw reconciliation failed: {}",
            String::from_utf8_lossy(&old.stderr)
        )
        .into());
    }
    let prior: Value = serde_json::from_slice(&old.stdout)?;
    let current = archived::audit(work, storage, changed)?;
    let p = data::protocol()?;
    for arm in ["original", "repeat"] {
        if read(&changed.join(arm).join("run/sealed/manifest.json"))?["identity"]["protocol_sha256"]
            != hash(data::RADII_PROTOCOL.as_bytes())
        {
            return Err("wrong learned-radius cohort".into());
        }
    }
    if prior["summary"]["measured"] != 64
        || current["summary"]["measured"] != 64
        || prior["summary"]["source_unavailable"] != 6
        || current["summary"]["unavailable_original_data"] != 6
    {
        return Err("changed original coverage".into());
    }
    let mut matched = BTreeMap::new();
    let mut initial_files = 0;
    for (id, s) in &p.cases {
        let (task, pins) = data::original(&p, id)?;
        let source = [full, prefix, rewrite]
            .iter()
            .map(|r| r.join("original/run").join(id))
            .find(|d| d.join("scores/report.json").is_file())
            .ok_or("missing previous model")?;
        let target = changed.join("original/run").join(id);
        let a = read(&source.join("scores/report.json"))?;
        let b = read(&target.join("scores/report.json"))?;
        let mut fits = BTreeMap::new();
        for mode in &p.modes {
            for seed in &p.seeds {
                let cell = format!("{mode}-{seed}");
                let initial = format!("{cell}-initial");
                let endpoint = format!("{cell}-final");
                let oa = read(&source.join(&cell).join("receipt.json"))?;
                let ob = read(&target.join(&cell).join("receipt.json"))?;
                let (u, v) = (&oa["training"], &ob["training"]);
                let extra = if mode == "forward" {
                    2 * s.vocabulary * p.updates
                } else {
                    0
                };
                if oa["data_fingerprints"] != pins
                    || ob["data_fingerprints"] != pins
                    || [
                        "before",
                        "batch_sha256",
                        "updates",
                        "examples",
                        "targets",
                        "min_visits",
                        "max_visits",
                    ]
                    .iter()
                    .any(|&f| u[f].is_null() || u[f] != v[f])
                    || v["forward_loss_queries"].as_u64().ok_or("queries")?
                        != u["forward_loss_queries"].as_u64().ok_or("old queries")? + extra as u64
                    || a["cells"][&initial] != b["cells"][&initial]
                {
                    return Err(format!("training inputs, initial results or coordinate visits differ for {id}/{cell}").into());
                }
                for split in task.splits.keys() {
                    let x = fs::read(raw_file(&source.join("scores"), &initial, split))?;
                    let y = fs::read(raw_file(&target.join("scores"), &initial, split))?;
                    if x != y {
                        return Err(
                            format!("initial raw logits differ for {id}/{cell}/{split}").into()
                        );
                    }
                    initial_files += 1;
                }
                let mut model: Model =
                    serde_json::from_slice(&fs::read(target.join(&cell).join("model.json"))?)?;
                let range = model.validate()?.radii;
                if range.len() != s.vocabulary {
                    return Err("missing learned magnitudes".into());
                }
                let radii = model.weights[range.clone()].to_vec();
                let samples = train::samples(&model, &task.splits["train"])?;
                if train::metrics(&model, &samples)? != v["after"] {
                    return Err("fitted training scores differ".into());
                }
                model.weights[range].fill(1.);
                fits.insert(
                    cell,
                    json!({"previous":split_counts(&a["cells"][&endpoint])?,
                "current":split_counts(&b["cells"][&endpoint])?,"radii":radii,
                "trained":v["after"],"train_with_unit_radii":train::metrics(&model,&samples)?}),
                );
            }
        }
        matched.insert(id, fits);
    }
    let mut changes = BTreeMap::new();
    for metric in ["original", "strict"] {
        let mut gains = vec![];
        let mut losses = vec![];
        for (id, new) in current["cases"].as_object().ok_or("original catalog")? {
            let old = prior["cases"].get(id).ok_or("missing old case")?;
            if old["current"].is_null() != new["current"].is_null()
                || old["control"] != new["catalog_control"]
                || (old["state"] == "source_unavailable") != (new["source_unavailable"] == true)
            {
                return Err("source status differs".into());
            }
            if old["current"][metric] != new["current"][metric] {
                if new["current"][metric] == true {
                    gains.push(id);
                } else {
                    losses.push(id);
                }
            }
        }
        changes.insert(metric, json!({"gained":gains,"lost":losses}));
    }
    serde_json::to_writer_pretty(
        std::io::stdout().lock(),
        &json!({"previous_summary":prior["summary"],
        "current_summary":current["summary"],"cases":current["cases"],"changes":changes,"matched_endpoints":matched,
        "identical_initial_score_files":initial_files}),
    )?;
    println!();
    Ok(())
}

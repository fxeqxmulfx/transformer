//! Compare final arithmetic predictions for the same original rows and batches
//! after fitting with either the physical depth or its log-odds coordinate.
use super::{archived::audit, decisions::Decisions};
use composed_retrieval::{Result, data};
use learned_retrieval::cohort::{raw_file, read};
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, path::Path};

pub fn compare(
    workspace: &Path,
    storage: &Path,
    physical: &Path,
    logodds: &Path,
    p: &data::Protocol,
    logodds_audit: &Value,
) -> Result<Value> {
    let old = audit(workspace, storage, physical)?;
    for arm in ["original", "repeat"] {
        let seal = read(&physical.join(arm).join("run/sealed/manifest.json"))?;
        let [a, b] = [physical, logodds].map(|root| read(&root.join(arm).join("inventory.json")));
        let (a, b) = (a?, b?);
        if seal["identity"]["protocol_sha256"]
            != baseline::hash(data::ARITHMETIC_PROTOCOL.as_bytes())
            || ["cases", "fits", "raw_bytes"]
                .iter()
                .any(|&key| a[key].is_null() || a[key] != b[key])
        {
            return Err("physical-depth protocol or arithmetic source inventory differs".into());
        }
    }
    let mut cases = BTreeMap::new();
    let mut totals = BTreeMap::<_, Decisions>::new();
    for (id, case) in &p.cases {
        let dirs = [physical, logodds].map(|root| root.join("original/run").join(id));
        let mut cells = BTreeMap::new();
        for mode in &p.modes {
            for seed in &p.seeds {
                let cell = format!("{mode}-{seed}");
                let [a, b] = dirs
                    .each_ref()
                    .map(|dir| read(&dir.join(&cell).join("receipt.json")));
                let (a, b) = (a?, b?);
                if ["case", "mode", "seed", "data_fingerprints"]
                    .iter()
                    .any(|&key| a[key].is_null() || a[key] != b[key])
                    || [
                        "batch_sha256",
                        "updates",
                        "examples",
                        "targets",
                        "min_visits",
                        "max_visits",
                        "forward_loss_queries",
                    ]
                    .iter()
                    .any(|&key| {
                        a["training"][key].is_null() || a["training"][key] != b["training"][key]
                    })
                {
                    return Err(
                        format!("unmatched source or training budget for {id}/{cell}").into(),
                    );
                }
                let mut splits = BTreeMap::new();
                for split in ["train", "validation"] {
                    let [a, b] = dirs.each_ref().map(|dir| {
                        fs::read(raw_file(
                            &dir.join("scores"),
                            &format!("{cell}-final"),
                            split,
                        ))
                    });
                    let mut d = Decisions::default();
                    d.add(&a?, &b?, case.vocabulary)?;
                    let total = totals.entry(split).or_default();
                    total.rows += d.rows;
                    total.different += d.different;
                    total.max_logit_delta = total.max_logit_delta.max(d.max_logit_delta);
                    splits.insert(split, d);
                }
                cells.insert(cell, splits);
            }
        }
        cases.insert(id, json!({"cells":cells, "physical":old["cases"][id]["current"], "logodds":logodds_audit["cases"][id]["current"]}));
    }
    Ok(json!({"final_predictions":totals, "cases":cases}))
}

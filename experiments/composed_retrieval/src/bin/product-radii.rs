//! Frozen training-only investigation of object magnitudes in the product
//! channel. No held-out logits or score reports are read by this command.
#[path = "product-radii/measure.rs"]
mod measure;
use baseline::hash;
use bilinear::guard;
use composed_retrieval::{Result, data, model::Model};
use learned_retrieval::cohort::{raw_file, read};
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, path::Path};

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let [workspace, storage, study, arm] = args.as_slice() else {
        return Err("expected WORKSPACE STORAGE OFFSET_51_OUTPUT original|repeat".into());
    };
    let arm = arm
        .to_str()
        .filter(|a| matches!(*a, "original" | "repeat"))
        .ok_or("unknown independent arm")?;
    let study = Path::new(study);
    guard::admit(1 << 30, Path::new(workspace), Path::new(storage))?;
    let root = study.join(arm).join("run");
    let seal = read(&root.join("sealed/manifest.json"))?;
    let identity = &seal["identity"];
    if identity["protocol_sha256"] != hash(data::OFFSET_PROTOCOL.as_bytes())
        || identity["producer"] != fs::read_to_string(study.join("producer.commit"))?.trim()
        || identity["binary_sha256"] != hash(&fs::read(study.join(arm).join("composed-retrieval"))?)
    {
        return Err("unrecognized source of frozen CM116 weights".into());
    }
    let p = data::protocol()?;
    let mut cases = BTreeMap::new();
    for (id, spec) in p
        .cases
        .iter()
        .filter(|(_, s)| s.source_fingerprints.is_some())
    {
        let (task, pins) = data::original(&p, id)?;
        let rows = task.splits["train"]
            .iter()
            .map(|e| {
                let ([a, q], [-100, y]) = (e.tokens.as_slice(), e.targets.as_slice()) else {
                    return Err("expected a masked pair".into());
                };
                Ok(measure::Row {
                    pair: [usize::try_from(*a)?, usize::try_from(*q)?],
                    target: usize::try_from(*y)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut cells = BTreeMap::new();
        for mode in &p.modes {
            for seed in &p.seeds {
                let cell = format!("{mode}-{seed}");
                let key = format!("{id}/{cell}");
                let receipt_bytes = fs::read(root.join(&key).join("receipt.json"))?;
                let receipt: Value = serde_json::from_slice(&receipt_bytes)?;
                let bytes = fs::read(root.join(&key).join("model.json"))?;
                if receipt["identity"] != *identity
                    || receipt["data_fingerprints"] != pins
                    || receipt["case"] != *id
                    || receipt["mode"] != *mode
                    || receipt["seed"] != *seed
                    || seal["cells"][&key] != hash(&receipt_bytes)
                    || receipt["endpoint_sha256"] != hash(&bytes)
                {
                    return Err("endpoint or original training source differs from seal".into());
                }
                let m: Model = serde_json::from_slice(&bytes)?;
                if m.vocabulary != spec.vocabulary || m.context != 2 {
                    return Err("invalid arithmetic model dimensions".into());
                }
                let raw = fs::read(raw_file(
                    &root.join(id).join("scores"),
                    &format!("{cell}-final"),
                    "train",
                ))?;
                cells.insert(
                    cell,
                    measure::measure(&m, &rows, &raw, &receipt["training"]["after"])?,
                );
            }
        }
        cases.insert(id, cells);
    }
    if cases.len() != 48 {
        return Err("all original arithmetic training cases are required".into());
    }
    serde_json::to_writer_pretty(
        std::io::stdout().lock(),
        &json!({"cases":cases,"model_producer":identity["producer"],"data_split":"train"}),
    )?;
    println!();
    Ok(())
}

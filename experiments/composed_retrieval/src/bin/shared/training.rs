use baseline::{
    data::Example,
    hash,
    metrics::{self, Predictor},
};
use bilinear::guard;
use composed_retrieval::{Result, data, model::Model, operator::Tables, train::Sample};
use learned_retrieval::cohort::{raw_file, read};
use serde_json::{Value, json};
use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

struct TrainingPredictor<'a> {
    model: &'a Model,
    tables: Tables,
    bytes: Option<RefCell<Vec<u8>>>,
}
impl Predictor for TrainingPredictor<'_> {
    fn vocab(&self) -> usize {
        self.model.vocabulary
    }
    fn logits(&self, prefix: &[i64]) -> Vec<f64> {
        let prefix = self
            .model
            .objects(prefix)
            .expect("validated original train prefix");
        let values = self
            .tables
            .logits(self.model, &self.tables.pass(self.model, &prefix).hidden);
        if let Some(bytes) = &self.bytes {
            bytes
                .borrow_mut()
                .extend(values.iter().flat_map(|x| x.to_le_bytes()));
        }
        values
    }
}

pub fn lengths(batch: &[&Sample], rows: &[Example]) -> Vec<usize> {
    let mut lengths: BTreeSet<_> = batch
        .iter()
        .map(|s| s.prefix.len().saturating_sub(1))
        .collect();
    for row in rows.iter().filter(|r| !r.prompt.is_empty()) {
        let n = row.prompt.len() - 1;
        lengths.extend(n..n + row.limit);
    }
    lengths.into_iter().collect()
}

/// Evaluate every supervised position and greedily regenerate each original
/// training answer. At the parent setting also reproduce the persisted bytes.
pub fn scores(
    m: &Model,
    lengths: &[usize],
    rows: &[Example],
    baseline: Option<(&[u8], &Value)>,
) -> Result<Value> {
    let predictor = TrainingPredictor {
        model: m,
        tables: Tables::new(m, lengths)?,
        bytes: baseline.map(|_| RefCell::default()),
    };
    let scores = metrics::evaluate(&predictor, rows)?;
    let teacher = scores.get("teacher_forced").unwrap_or(&scores);
    if let Some((raw, after)) = baseline {
        let ce = after["ce"].as_f64().ok_or("receipt CE")?;
        if predictor.bytes.unwrap().into_inner() != raw
            || teacher["correct"] != after["correct"]
            || teacher["targets"] != after["targets"]
            || (teacher["loss"].as_f64().ok_or("training CE")? - ce).abs() > 1e-12
        {
            return Err("zero intervention differs from sealed training vectors or receipt".into());
        }
    }
    Ok(
        json!({"ce":teacher["loss"],"teacher_correct":teacher["correct"],"correct":scores["correct"],
        "exact":scores["exact"],"targets":scores["targets"],"examples":scores["examples"]}),
    )
}

/// The six original Rewrite families and two-hop lookup, all frozen training
/// endpoints, with only original training records passed to the intervention.
pub fn rewrite_sweep(
    producer: &str,
    protocol: &str,
    intervention: fn(&Model, &[Example], &[u8], &Value) -> Result<Value>,
) -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let [workspace, storage, study, arm] = args.as_slice() else {
        return Err("expected WORKSPACE STORAGE FROZEN_STUDY original|repeat".into());
    };
    let arm = arm
        .to_str()
        .filter(|x| matches!(*x, "original" | "repeat"))
        .ok_or("unknown arm")?;
    let study = Path::new(study);
    guard::admit(1 << 30, Path::new(workspace), Path::new(storage))?;
    let root = study.join(arm).join("run");
    let seal = read(&root.join("sealed/manifest.json"))?;
    let identity = &seal["identity"];
    if data::PROTOCOL != protocol
        || identity["producer"] != producer
        || fs::read_to_string(study.join("producer.commit"))?.trim() != producer
        || identity["protocol_sha256"] != hash(protocol.as_bytes())
        || identity["binary_sha256"] != hash(&fs::read(study.join(arm).join("composed-retrieval"))?)
    {
        return Err("unrecognized frozen study".into());
    }
    let p = data::protocol()?;
    let mut cases = BTreeMap::new();
    for id in synthetic_data::generative::CASES
        .into_iter()
        .chain(["suite-lookup"])
    {
        let spec = p.cases.get(id).ok_or("missing original case")?;
        let (source, pins) = data::original(&p, id)?;
        let mut cells = BTreeMap::new();
        for mode in &p.modes {
            for seed in &p.seeds {
                let cell = format!("{mode}-{seed}");
                let key = format!("{id}/{cell}");
                let bytes = fs::read(root.join(&key).join("receipt.json"))?;
                let receipt: Value = serde_json::from_slice(&bytes)?;
                let weights = fs::read(root.join(&key).join("model.json"))?;
                if seal["cells"][&key] != hash(&bytes)
                    || receipt["data_fingerprints"] != pins
                    || receipt["endpoint_sha256"] != hash(&weights)
                    || receipt["identity"] != *identity
                    || receipt["case"] != id
                    || receipt["mode"] != *mode
                    || receipt["seed"] != *seed
                {
                    return Err("training receipt or endpoint differs".into());
                }
                let m: Model = serde_json::from_slice(&weights)?;
                if m.context != spec.context || m.vocabulary != spec.vocabulary {
                    return Err("incompatible model dimensions".into());
                }
                let raw = fs::read(raw_file(
                    &root.join(id).join("scores"),
                    &format!("{cell}-final"),
                    "train",
                ))?;
                cells.insert(
                    cell,
                    intervention(
                        &m,
                        &source.splits["train"],
                        &raw,
                        &receipt["training"]["after"],
                    )?,
                );
            }
        }
        cases.insert(id, cells);
    }
    serde_json::to_writer_pretty(
        std::io::stdout().lock(),
        &json!({"model_producer":producer,"data_split":"train","cases":cases}),
    )?;
    println!();
    Ok(())
}

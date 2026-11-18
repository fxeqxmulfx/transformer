//! Training-only CM118 intervention on the complete frozen CM116 Rewrite
//! cohort. Reload original objects and compare every baseline training logit.
use baseline::{
    data::Example,
    hash,
    metrics::{self, Predictor},
};
use bilinear::guard;
use composed_retrieval::{Result, data, model::Model, operator::Tables, train};
use learned_retrieval::cohort::{raw_file, read};
use serde_json::{Value, json};
use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

const PRODUCER: &str = "9a222cfdd9c7364a864749f90dfc5d89d1d86a3f";
const GRID: [f64; 7] = [0., 0.0625, 0.125, 0.25, 0.5, 1., 2.];

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

fn intervention(m: &Model, rows: &[Example], raw: &[u8], after: &Value) -> Result<Value> {
    let layout = m.validate()?;
    let samples = train::samples(m, rows)?;
    let batch: Vec<_> = samples.iter().flatten().collect();
    let mut lengths: BTreeSet<_> = batch
        .iter()
        .map(|s| s.prefix.len().saturating_sub(1))
        .collect();
    let reach = lengths.last().ok_or("empty supervision")?.saturating_sub(1);
    for row in rows.iter().filter(|r| !r.prompt.is_empty()) {
        let n = row.prompt.len() - 1;
        lengths.extend(n..n + row.limit);
    }
    let lengths: Vec<_> = lengths.into_iter().collect();
    let residual = &m.weights[layout.relation.clone()];
    let center = residual.len() / 2;
    let unseen = residual
        .iter()
        .enumerate()
        .filter(|(i, _)| i.abs_diff(center) > reach);
    if unseen.clone().any(|(_, &v)| v != 0.) {
        return Err("an unobservable residual moved from zero".into());
    }
    let mut peaks: Vec<_> = residual
        .iter()
        .copied()
        .enumerate()
        .filter(|(i, _)| i.abs_diff(center) <= reach)
        .map(|(i, v)| (i as i64 - center as i64, v))
        .collect();
    peaks.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    peaks.truncate(3);
    let tables = Tables::new(m, &lengths)?;
    let gradient = train::analytic(m, &tables, &batch);
    let slope = -gradient[layout.relation.clone()]
        .iter()
        .enumerate()
        .map(|(i, g)| i.abs_diff(center) as f64 * g)
        .sum::<f64>();
    let with_decay = |decay: f64| {
        let mut perturbed = m.clone();
        for (i, value) in perturbed.weights[layout.relation.clone()]
            .iter_mut()
            .enumerate()
        {
            *value -= decay * i.abs_diff(center) as f64;
        }
        perturbed
    };
    let step = 1. / 65536.;
    let loss = |a| -> Result<f64> {
        train::metrics(&with_decay(a), &samples)?["ce"]
            .as_f64()
            .ok_or_else(|| "missing CE".into())
    };
    let origin = after["ce"].as_f64().ok_or("receipt CE")?;
    let numerical = (4. * (loss(step)? - origin) - (loss(2. * step)? - origin)) / (2. * step);
    if (slope - numerical).abs() > 2e-7 * (1. + slope.abs()) {
        return Err(format!(
            "distance derivative differs from feasible queries: {slope} vs {numerical}"
        )
        .into());
    }
    let mut grid = Vec::new();
    for decay in GRID {
        let perturbed = with_decay(decay);
        let predictor = TrainingPredictor {
            model: &perturbed,
            tables: Tables::new(&perturbed, &lengths)?,
            bytes: (decay == 0.).then(RefCell::default),
        };
        let scores = metrics::evaluate(&predictor, rows)?;
        let teacher = scores.get("teacher_forced").unwrap_or(&scores);
        if let Some(bytes) = predictor.bytes
            && (bytes.into_inner() != raw
                || teacher["correct"] != after["correct"]
                || teacher["targets"] != after["targets"]
                || (teacher["loss"].as_f64().ok_or("training CE")? - origin).abs() > 1e-12)
        {
            return Err("zero decay differs from sealed training vectors or receipt".into());
        }
        grid.push(json!({"decay":decay,"ce":teacher["loss"],"teacher_correct":teacher["correct"],
            "correct":scores["correct"],"exact":scores["exact"],"targets":scores["targets"],"examples":scores["examples"]}));
    }
    Ok(
        json!({"max_observed_distance":reach,"unobserved_residuals":unseen.count(),
        "peak_observed_residuals":peaks,"ce_right_derivative_at_zero":slope,"forward_queried_derivative":numerical,"training":grid}),
    )
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let [workspace, storage, study, arm] = args.as_slice() else {
        return Err("expected WORKSPACE STORAGE CM116_REWRITE_OUTPUT original|repeat".into());
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
    if data::PROTOCOL != data::REWRITE_PROTOCOL
        || identity["producer"] != PRODUCER
        || fs::read_to_string(study.join("producer.commit"))?.trim() != PRODUCER
        || identity["protocol_sha256"] != hash(data::REWRITE_PROTOCOL.as_bytes())
        || identity["binary_sha256"] != hash(&fs::read(study.join(arm).join("composed-retrieval"))?)
    {
        return Err("unrecognized frozen Rewrite study".into());
    }
    let p = data::protocol()?;
    let mut cases = BTreeMap::new();
    for (id, spec) in &p.cases {
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
                    || receipt["case"] != *id
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
        &json!({"model_producer":PRODUCER,"data_split":"train","cases":cases}),
    )?;
    println!();
    Ok(())
}

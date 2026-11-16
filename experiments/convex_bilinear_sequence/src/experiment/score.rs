//! Exhaustive original metrics with a separate, complete raw-logit replay.
//! Every declared training endpoint must be sealed before any scoring.
use super::*;
use crate::{algebra, guard, model::Joint};
use baseline::metrics::Predictor;
use serde_json::json;
use std::{
    cell::RefCell,
    fs,
    path::{Path, PathBuf},
};

pub fn cell(cohort: &Path, name: &str, mode: &str, seed: u64) -> PathBuf {
    cohort.join(name).join(mode).join(format!("seed{seed}"))
}

pub fn read(path: &Path) -> Result<Value> {
    Ok(serde_json::from_reader(std::io::BufReader::new(
        fs::File::open(path)?,
    ))?)
}

pub fn sealed(cohort: &Path) -> Result<()> {
    sealed_with_protocol(cohort, PROTOCOL)
}

pub fn sealed_with_protocol(cohort: &Path, protocol_text: &str) -> Result<()> {
    let protocol: Value = serde_json::from_str(protocol_text)?;
    for config in protocol["cases"].as_array().ok_or("missing cases")? {
        let name = config["name"].as_str().ok_or("case name")?;
        for mode in protocol["modes"].as_array().ok_or("missing modes")? {
            for seed in protocol["seeds"].as_array().ok_or("missing seeds")? {
                let path = cell(
                    cohort,
                    name,
                    mode.as_str().ok_or("mode")?,
                    seed.as_u64().ok_or("seed")?,
                );
                let receipt = read(&path.join("training_receipt.json"))?;
                let meta = read(&path.join("metadata.json"))?;
                if receipt["status"] != "sealed"
                    || receipt["heldout_scored"] != false
                    || receipt["source_fingerprint"] != fingerprint()
                    || meta["source_fingerprint"] != fingerprint()
                    || meta["protocol"] != protocol
                    || receipt["updates"] != config["maximum_updates"]
                    || receipt["case"] != name
                    || receipt["mode"] != *mode
                    || receipt["seed"] != *seed
                    || meta["case"] != *config
                    || meta["mode"] != *mode
                    || meta["seed"] != *seed
                    || !path.join("endpoint.json").is_file()
                {
                    return Err(format!("cohort seal refused: {}", path.display()).into());
                }
            }
        }
    }
    Ok(())
}

struct AuditScorer<'a> {
    model: &'a Model,
    digest: RefCell<Sha256>,
    calls: RefCell<u64>,
}
impl Predictor for AuditScorer<'_> {
    fn vocab(&self) -> usize {
        self.model.joint.shape.objects
    }
    fn logits(&self, prefix: &[i64]) -> Vec<f64> {
        let logits = self
            .model
            .logits(prefix)
            .unwrap_or_else(|error| panic!("raw replay refused: {error}"));
        let mut digest = self.digest.borrow_mut();
        digest.update((prefix.len() as u64).to_le_bytes());
        for token in prefix {
            digest.update(token.to_le_bytes());
        }
        digest.update((logits.len() as u64).to_le_bytes());
        for value in &logits {
            digest.update(value.to_bits().to_le_bytes());
        }
        *self.calls.borrow_mut() += 1;
        logits
    }
}

pub fn evaluate(model: &Model, examples: &[Example]) -> Result<Value> {
    // The initial cohort has no residue-pooled or numeric-output metrics.
    // Refuse those here until their separate original/raw fields are ported.
    if examples.iter().any(|e| e.field.is_some()) {
        return Err("raw field metric needs a separate port".into());
    }
    let scorer = AuditScorer {
        model,
        digest: RefCell::new(Sha256::new()),
        calls: RefCell::new(0),
    };
    let metrics = baseline::metrics::evaluate(&scorer, examples)?;
    let digest: String = scorer
        .digest
        .into_inner()
        .finalize()
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect();
    Ok(
        json!({"metrics":metrics,"raw_logits_sha256":digest,"forward_calls":scorer.calls.into_inner()}),
    )
}

pub fn run(
    workspace: &Path,
    storage: &Path,
    cohort: &Path,
    name: &str,
    mode_name: &str,
    seed: u64,
    (protocol_text, replay): (&str, bool),
) -> Result<()> {
    guard::admit(0, workspace, storage)?;
    sealed_with_protocol(cohort, protocol_text)?;
    let protocol: Value = serde_json::from_str(protocol_text)?;
    let (_, original, task) = case_in(name, protocol_text)?;
    mode(mode_name)?;
    if !protocol["seeds"]
        .as_array()
        .ok_or("seeds")?
        .contains(&json!(seed))
    {
        return Err("seed outside cohort".into());
    }
    let path = cell(cohort, name, mode_name, seed);
    let receipt = read(&path.join("training_receipt.json"))?;
    let meta = read(&path.join("metadata.json"))?;
    let shape = shape(&task)?;
    if meta["original"] != original
        || meta["shape"] != serde_json::to_value(shape)?
        || meta["selection"] != task.selection
    {
        return Err("checkpoint dataset identity changed".into());
    }
    let endpoint = path.join("endpoint.json");
    let bytes = fs::metadata(&endpoint)?.len();
    let columns = usize::try_from(receipt["stored_columns"].as_u64().ok_or("stored columns")?)?;
    guard::admit(
        shape
            .reservation(columns)?
            .checked_add(bytes.checked_mul(3).ok_or("checkpoint size overflow")?)
            .ok_or("checkpoint reservation overflow")?,
        workspace,
        storage,
    )?;
    let raw = fs::read(&endpoint)?;
    let endpoint_hash = hash(&raw);
    if receipt["endpoint_sha256"] != endpoint_hash {
        return Err("sealed checkpoint hash mismatch".into());
    }
    let joint: Joint = serde_json::from_slice(&raw)?;
    drop(raw);
    if joint.shape != shape || joint.columns.len() != columns {
        return Err("checkpoint domain mismatch".into());
    }
    algebra::set_threads(protocol["blas_threads"].as_i64().ok_or("threads")? as i32)?;
    let model = Model::recover(
        joint,
        protocol["model_reservation_bytes"]
            .as_u64()
            .ok_or("budget")?,
    )?;
    let mut splits = serde_json::Map::new();
    for (split, examples) in &task.splits {
        guard::admit(0, workspace, storage)?;
        let report = evaluate(&model, examples)?;
        println!(
            "case={name} mode={mode_name} split={split} raw={} exact={} ce={}",
            report["metrics"]["token_accuracy"],
            report["metrics"]["sequence_accuracy"],
            report["metrics"]["loss"]
        );
        splits.insert(split.clone(), report);
    }
    let selected = &splits[&task.selection]["metrics"];
    let metric = original["case"]["metric"].as_str().ok_or("source metric")?;
    let threshold = original["case"]["target"].as_f64().ok_or("source target")?;
    let success = selected[metric].as_f64().ok_or("missing source metric")? >= threshold;
    let raw99 = selected["token_accuracy"].as_f64().ok_or("raw accuracy")? >= 0.99
        && selected["sequence_accuracy"]
            .as_f64()
            .ok_or("exact accuracy")?
            >= 0.99;
    let result = json!({"source_fingerprint":fingerprint(),"endpoint_sha256":endpoint_hash,
        "case":name,"mode":mode_name,"seed":seed,"selection":task.selection,
        "original_success":success,"raw_and_exact_99":raw99,"original_metric":metric,
        "original_threshold":threshold,"splits":splits});
    if replay {
        let saved = read(&path.join("scores.json"))?;
        if saved != result {
            return Err("complete metrics/raw-logit replay mismatch".into());
        }
        write_json(
            &path.join("replay.json"),
            &json!({"status":"identical","source_fingerprint":fingerprint(),
            "score_sha256":hash(&fs::read(path.join("scores.json"))?),"endpoint_sha256":endpoint_hash}),
        )?;
    } else {
        write_json(&path.join("scores.json"), &result)?;
    }
    Ok(())
}

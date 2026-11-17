//! Complete original metrics and separate-process raw-logit replay after sealing.
//! Ported from shared-atoms 41cffd66; check complete scalar parameters and budget.
use super::{PROTOCOL, fingerprint, initial_model, protocol};
use crate::{Result, model::Model};
use baseline::{data::Example, metrics::Predictor};
use bilinear::{experiment as original, guard};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{cell::RefCell, fs, path::Path};

pub fn sealed(cohort: &Path) -> Result<()> {
    sealed_with_protocol(cohort, PROTOCOL)
}

pub fn sealed_with_protocol(cohort: &Path, text: &str) -> Result<()> {
    let protocol: Value = serde_json::from_str(text)?;
    let binary = original::hash(&fs::read(std::env::current_exe()?)?);
    for case in protocol["cases"].as_array().ok_or("cases")? {
        let name = case["name"].as_str().ok_or("case")?;
        for mode in protocol["modes"].as_array().ok_or("modes")? {
            for seed in protocol["seeds"].as_array().ok_or("seeds")? {
                let path = original::score::cell(
                    cohort,
                    name,
                    mode.as_str().ok_or("mode")?,
                    seed.as_u64().ok_or("seed")?,
                );
                let receipt = original::score::read(&path.join("training_receipt.json"))?;
                let meta = original::score::read(&path.join("metadata.json"))?;
                if receipt["status"] != "sealed"
                    || receipt["heldout_scored"] != false
                    || receipt["source_fingerprint"] != fingerprint()
                    || meta["source_fingerprint"] != fingerprint()
                    || meta["protocol"] != protocol
                    || meta["case"] != *case
                    || meta["binary_sha256"] != binary
                    || receipt["updates"] != case["maximum_updates"]
                    || receipt["case"] != name
                    || receipt["mode"] != *mode
                    || receipt["seed"] != *seed
                    || meta["mode"] != *mode
                    || meta["seed"] != *seed
                    || !path.join("endpoint.json").is_file()
                    || receipt["endpoint_sha256"]
                        != original::hash(&fs::read(path.join("endpoint.json"))?)
                {
                    return Err(format!("cohort seal refused: {}", path.display()).into());
                }
            }
        }
    }
    Ok(())
}

struct Scorer<'a> {
    model: &'a Model,
    digest: RefCell<Sha256>,
    calls: RefCell<usize>,
}
impl Predictor for Scorer<'_> {
    fn vocab(&self) -> usize {
        self.model.objects
    }
    fn logits(&self, prefix: &[i64]) -> Vec<f64> {
        let logits = self
            .model
            .logits(prefix)
            .unwrap_or_else(|e| panic!("complete forward refused: {e}"));
        let mut digest = self.digest.borrow_mut();
        digest.update((prefix.len() as u64).to_le_bytes());
        for t in prefix {
            digest.update(t.to_le_bytes());
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
    if examples.iter().any(|e| e.field.is_some()) {
        return Err("raw field metrics require a separate port".into());
    }
    let scorer = Scorer {
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
    mode: &str,
    seed: u64,
    options: (&str, bool),
) -> Result<()> {
    let (text, replay) = options;
    guard::admit(0, workspace, storage)?;
    let (protocol, _) = protocol(mode, seed, text)?;
    sealed_with_protocol(cohort, text)?;
    let (_, original_row, task) = original::case_in(name, text)?;
    let path = original::score::cell(cohort, name, mode, seed);
    let meta = original::score::read(&path.join("metadata.json"))?;
    let receipt = original::score::read(&path.join("training_receipt.json"))?;
    let expected = initial_model(task.vocab, &protocol)?;
    if meta["original"] != original_row
        || meta["objects"] != json!(task.vocab)
        || meta["selection"] != task.selection
    {
        return Err("original dataset identity changed".into());
    }
    let endpoint = path.join("endpoint.json");
    guard::admit(
        fs::metadata(&endpoint)?
            .len()
            .checked_mul(5)
            .ok_or("checkpoint reservation overflow")?,
        workspace,
        storage,
    )?;
    let raw = fs::read(&endpoint)?;
    let hash = original::hash(&raw);
    if receipt["endpoint_sha256"] != hash {
        return Err("sealed checkpoint hash mismatch".into());
    }
    let model: Model = serde_json::from_slice(&raw)?;
    drop(raw);
    if model.objects != task.vocab
        || meta["budget"] != json!(model.budget)
        || model.budget != expected.budget
        || meta["real_coordinates_per_component"] != json!(2 * (1 + 2 * task.vocab))
        || receipt["components_stored"] != json!(model.atoms.len())
    {
        return Err("checkpoint capacity/domain mismatch".into());
    }
    model.validate()?;
    let mut splits = serde_json::Map::new();
    for (name, examples) in &task.splits {
        guard::admit(0, workspace, storage)?;
        let report = evaluate(&model, examples)?;
        println!(
            "mode={mode} split={name} raw={} exact={} ce={}",
            report["metrics"]["token_accuracy"],
            report["metrics"]["sequence_accuracy"],
            report["metrics"]["loss"]
        );
        splits.insert(name.clone(), report);
    }
    let metric = original_row["case"]["metric"]
        .as_str()
        .ok_or("source metric")?;
    let threshold = original_row["case"]["target"]
        .as_f64()
        .ok_or("source target")?;
    let selected = &splits[&task.selection]["metrics"];
    let success = selected[metric].as_f64().ok_or("source metric result")? >= threshold;
    let raw99 = selected["token_accuracy"].as_f64().ok_or("raw accuracy")? >= 0.99
        && selected["sequence_accuracy"]
            .as_f64()
            .ok_or("exact accuracy")?
            >= 0.99;
    let result = json!({"source_fingerprint":fingerprint(),"endpoint_sha256":hash,"case":name,"mode":mode,
        "seed":seed,"selection":task.selection,"original_success":success,"raw_and_exact_99":raw99,
        "original_metric":metric,"original_threshold":threshold,"splits":splits});
    if replay {
        if result != original::score::read(&path.join("scores.json"))? {
            return Err("complete raw-logit/metric replay mismatch".into());
        }
        original::write_json(
            &path.join("replay.json"),
            &json!({"status":"identical","source_fingerprint":fingerprint(),
            "endpoint_sha256":hash,"score_sha256":original::hash(&fs::read(path.join("scores.json"))?)}),
        )?;
    } else {
        original::write_json(&path.join("scores.json"), &result)?;
    }
    Ok(())
}

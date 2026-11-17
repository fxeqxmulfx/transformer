//! Complete original scoring only after every prospective endpoint is sealed.
use crate::{Result, experiment, model::Model, source};
use baseline::{data::Example, metrics::Predictor};
use bilinear::{experiment as original, guard};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{cell::RefCell, fs, path::Path};
pub fn sealed(cohort: &Path) -> Result<()> {
    let binary = original::hash(&fs::read(std::env::current_exe()?)?);
    for arm in experiment::ARMS {
        for variant in experiment::VARIANTS {
            for mode in experiment::MODES {
                let (p, _, _) = experiment::settings(arm, variant, mode)?;
                let path = experiment::cell(cohort, arm, variant, mode);
                let m = original::score::read(&path.join("metadata.json"))?;
                let r = original::score::read(&path.join("training_receipt.json"))?;
                if m["source_fingerprint"] != experiment::fingerprint()
                    || r["source_fingerprint"] != experiment::fingerprint()
                    || m["binary_sha256"] != binary
                    || m["protocol"] != p
                    || m["input"] != source::pin(variant, mode)?
                    || m["arm"] != arm
                    || m["variant"] != variant
                    || m["mode"] != mode
                    || r["arm"] != arm
                    || r["variant"] != variant
                    || r["mode"] != mode
                    || r["updates"] != p["updates"]
                    || r["status"] != "sealed"
                    || r["heldout_scored"] != false
                    || r["parent_sha256"] != m["parent_sha256"]
                {
                    return Err("incomplete or changed Hilbert cohort seal".into());
                }
                for (file, key) in [
                    ("endpoint.json", "endpoint_sha256"),
                    ("cache.json", "cache_sha256"),
                    ("history.jsonl", "history_sha256"),
                    ("training.jsonl", "trace_sha256"),
                ] {
                    if r[key] != original::hash(&fs::read(path.join(file))?) {
                        return Err("changed sealed Hilbert artifact".into());
                    }
                }
            }
        }
    }
    Ok(())
}
pub fn load(
    workspace: &Path,
    storage: &Path,
    path: &Path,
) -> Result<(Model, Value, baseline::data::Task)> {
    let meta = original::score::read(&path.join("metadata.json"))?;
    let arm = meta["arm"].as_str().ok_or("arm")?;
    let variant = meta["variant"].as_str().ok_or("variant")?;
    let mode = meta["mode"].as_str().ok_or("mode")?;
    let (p, geometry, _) = experiment::settings(arm, variant, mode)?;
    let (parent, _, task) = source::load(workspace, storage, variant, mode)?;
    let (_, row, _) = original::case_in(
        readouts::source::CASE,
        rotary::experiment::CAUTIOUS_PROTOCOL,
    )?;
    guard::admit(
        fs::metadata(path.join("endpoint.json"))?
            .len()
            .checked_mul(8)
            .ok_or("score reservation")?,
        workspace,
        storage,
    )?;
    let model: Model = serde_json::from_slice(&fs::read(path.join("endpoint.json"))?)?;
    let parent_bytes = serde_json::to_vec(&parent)?;
    if meta["original"] != row
        || meta["selection"] != task.selection
        || meta["objects"] != task.vocab
        || meta["budget"] != model.parent.budget
        || model.block.geometry != geometry
        || serde_json::to_vec(&model.parent)? != parent_bytes
        || meta["parent_sha256"] != original::hash(&parent_bytes)
        || p["new_mass_fraction"] != model.fraction
        || meta["components_stored"] != model.parent.atoms.len() + 1
        || meta["active_real_coordinates"] != model.block.dimensions()
    {
        return Err("complete mixed family or original scoring identity changed".into());
    }
    model.validate()?;
    Ok((model, row, task))
}
struct Scorer<'a> {
    model: &'a Model,
    digest: RefCell<Sha256>,
    calls: RefCell<usize>,
}
impl Predictor for Scorer<'_> {
    fn vocab(&self) -> usize {
        self.model.parent.objects
    }
    fn logits(&self, prefix: &[i64]) -> Vec<f64> {
        let logits = self
            .model
            .logits(prefix)
            .unwrap_or_else(|e| panic!("complete mixed forward refused: {e}"));
        let mut digest = self.digest.borrow_mut();
        digest.update((prefix.len() as u64).to_le_bytes());
        for t in prefix {
            digest.update(t.to_le_bytes());
        }
        digest.update((logits.len() as u64).to_le_bytes());
        for x in &logits {
            digest.update(x.to_bits().to_le_bytes());
        }
        *self.calls.borrow_mut() += 1;
        logits
    }
}
pub fn evaluate(model: &Model, examples: &[Example]) -> Result<Value> {
    if examples.iter().any(|e| e.field.is_some()) {
        return Err("field metric requires a separate original port".into());
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
    arm: &str,
    variant: &str,
    mode: &str,
    replay: bool,
) -> Result<()> {
    guard::admit(0, workspace, storage)?;
    sealed(cohort)?;
    experiment::settings(arm, variant, mode)?;
    let path = experiment::cell(cohort, arm, variant, mode);
    let (model, row, task) = load(workspace, storage, &path)?;
    let mut splits = serde_json::Map::new();
    for (name, examples) in &task.splits {
        guard::admit(0, workspace, storage)?;
        let mut r = evaluate(&model, examples)?;
        r["raw_and_exact_99"] = json!(
            r["metrics"]["token_accuracy"].as_f64().ok_or("raw")? >= 0.99
                && r["metrics"]["sequence_accuracy"].as_f64().ok_or("exact")? >= 0.99
        );
        println!(
            "{arm}/{variant}/{mode} split={name} raw={} exact={} ce={}",
            r["metrics"]["token_accuracy"], r["metrics"]["sequence_accuracy"], r["metrics"]["loss"]
        );
        splits.insert(name.clone(), r);
    }
    let metric = row["case"]["metric"].as_str().ok_or("metric")?;
    let target = row["case"]["target"].as_f64().ok_or("target")?;
    let report = json!({"source_fingerprint":experiment::fingerprint(),"endpoint_sha256":original::hash(&fs::read(path.join("endpoint.json"))?),"arm":arm,"variant":variant,"mode":mode,"selection":task.selection,"original_metric":metric,"original_threshold":target,"original_success":splits[&task.selection]["metrics"][metric].as_f64().ok_or("primary")?>=target,"raw_and_exact_99":splits[&task.selection]["raw_and_exact_99"],"splits":splits});
    if replay {
        if report != original::score::read(&path.join("scores.json"))? {
            return Err("complete mixed raw replay differs".into());
        }
        original::write_json(
            &path.join("replay.json"),
            &json!({"status":"identical","source_fingerprint":experiment::fingerprint(),"score_sha256":original::hash(&fs::read(path.join("scores.json"))?)}),
        )?;
    } else {
        original::write_json(&path.join("scores.json"), &report)?;
    }
    Ok(())
}

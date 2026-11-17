//! Score all declared checkpoints only after the complete prospective cohort seals.
use crate::{Result, experiment, model::Model, seal, source};
use baseline::{data::Example, metrics::Predictor};
use bilinear::{experiment as original, guard};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{cell::RefCell, fs, path::Path};
struct Scorer<'a> {
    model: &'a Model,
    digest: RefCell<Sha256>,
    calls: RefCell<usize>,
}
impl Predictor for Scorer<'_> {
    fn vocab(&self) -> usize {
        self.model.objects()
    }
    fn logits(&self, prefix: &[i64]) -> Vec<f64> {
        let logits = self
            .model
            .logits(prefix)
            .unwrap_or_else(|e| panic!("complete mixture refused: {e}"));
        let mut h = self.digest.borrow_mut();
        h.update((prefix.len() as u64).to_le_bytes());
        for t in prefix {
            h.update(t.to_le_bytes());
        }
        h.update((logits.len() as u64).to_le_bytes());
        for x in &logits {
            h.update(x.to_bits().to_le_bytes());
        }
        *self.calls.borrow_mut() += 1;
        logits
    }
}
pub fn evaluate(model: &Model, examples: &[Example]) -> Result<Value> {
    if examples.iter().any(|e| e.field.is_some()) {
        return Err("field metric needs its original port".into());
    }
    let s = Scorer {
        model,
        digest: RefCell::new(Sha256::new()),
        calls: RefCell::new(0),
    };
    let metrics = baseline::metrics::evaluate(&s, examples)?;
    let digest: String = s
        .digest
        .into_inner()
        .finalize()
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect();
    Ok(json!({"metrics":metrics,"raw_logits_sha256":digest,"forward_calls":s.calls.into_inner()}))
}
pub fn run(workspace: &Path, storage: &Path, cohort: &Path, replay: bool) -> Result<()> {
    seal::all(workspace, storage, cohort)?;
    let p = experiment::protocol()?;
    for cell in experiment::cells() {
        let (mut model, row, task) = seal::initial(workspace, storage, cohort, &cell)?;
        let path = cell.path(cohort);
        let mut checkpoints = serde_json::Map::new();
        for n in 0..=p["stages"].as_u64().ok_or("stages")? as usize {
            if n > 0 {
                let active = experiment::prepare(&mut model, &cell, n, &p)?;
                model = seal::transition(&model, &experiment::stage(&path, n), active)?;
            }
            if !p["checkpoints"]
                .as_array()
                .ok_or("checkpoints")?
                .contains(&json!(n))
            {
                continue;
            }
            let mut splits = serde_json::Map::new();
            for (name, examples) in &task.splits {
                guard::admit(0, workspace, storage)?;
                let mut r = evaluate(&model, examples)?;
                r["raw_and_exact_99"] = json!(
                    r["metrics"]["token_accuracy"].as_f64().ok_or("raw")? >= 0.99
                        && r["metrics"]["sequence_accuracy"].as_f64().ok_or("exact")? >= 0.99
                );
                splits.insert(name.clone(), r);
            }
            if n == 0
                && json!(splits)
                    != original::score::read(&source::path(storage, &cell).join("scores.json"))?["splits"]
            {
                return Err("initial complete raw report changed".into());
            }
            let metric = row["case"]["metric"].as_str().ok_or("metric")?;
            let target = row["case"]["target"].as_f64().ok_or("target")?;
            let r = json!({"model_sha256":original::hash(&serde_json::to_vec(&model)?),"components_stored":model.components(),"capacity_mass":model.used_mass(),"original_metric":metric,"original_threshold":target,"original_success":splits[&task.selection]["metrics"][metric].as_f64().ok_or("primary")?>=target,"raw_and_exact_99":splits[&task.selection]["raw_and_exact_99"],"splits":splits});
            println!(
                "{} checkpoint={n} train={} novel={} test={}",
                cell.label(),
                r["splits"]["train"]["metrics"]["token_accuracy"],
                r["splits"]["validation/novel"]["metrics"]["token_accuracy"],
                r["splits"]["test"]["metrics"]["token_accuracy"]
            );
            checkpoints.insert(n.to_string(), r);
        }
        let report = json!({"source_fingerprint":experiment::fingerprint(),"cell":cell,"selection":task.selection,"checkpoints":checkpoints});
        if replay {
            if report != original::score::read(&path.join("scores.json"))? {
                return Err("complete growth raw replay differs".into());
            }
            original::write_json(
                &path.join("replay.json"),
                &json!({"status":"identical","source_fingerprint":experiment::fingerprint(),"score_sha256":original::hash(&fs::read(path.join("scores.json"))?)}),
            )?;
        } else {
            original::write_json(&path.join("scores.json"), &report)?;
        }
    }
    Ok(())
}

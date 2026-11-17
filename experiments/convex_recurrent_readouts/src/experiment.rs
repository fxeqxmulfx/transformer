//! Prospective complete refit cohort; every endpoint seals before original scoring.
use crate::{
    Result, continuation, source,
    update::{self, Config},
};
use baseline::random::Random;
use bilinear::{experiment as original, guard};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};
pub const PROTOCOL: &str = include_str!("../protocol.json");
pub const LONG_PROTOCOL: &str = include_str!("../long_protocol.json");
pub const VARIANTS: [&str; 2] = ["cautious", "refined"];
pub const MODES: [&str; 2] = ["mini", "forward"];
pub fn fingerprint() -> String {
    let mut h = Sha256::new();
    for bytes in [
        include_bytes!("lib.rs").as_slice(),
        include_bytes!("update.rs"),
        include_bytes!("source.rs"),
        include_bytes!("continuation.rs"),
        include_bytes!("experiment.rs"),
        include_bytes!("score.rs"),
        include_bytes!("bin/stress.rs"),
        include_bytes!("../Cargo.toml"),
        include_bytes!("../Cargo.lock"),
        PROTOCOL.as_bytes(),
        LONG_PROTOCOL.as_bytes(),
        rotary::experiment::fingerprint().as_bytes(),
    ] {
        h.update((bytes.len() as u64).to_le_bytes());
        h.update(bytes);
    }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}
pub fn cell(cohort: &Path, variant: &str, mode: &str) -> PathBuf {
    cohort.join(variant).join(mode)
}
pub fn settings(variant: &str, mode: &str) -> Result<(Value, Config)> {
    settings_with(variant, mode, PROTOCOL)
}
pub fn settings_with(variant: &str, mode: &str, text: &str) -> Result<(Value, Config)> {
    if !VARIANTS.contains(&variant) || !MODES.contains(&mode) {
        return Err("cell outside frozen cohort".into());
    }
    let p: Value = serde_json::from_str(text)?;
    let config = Config {
        mode: original::mode(mode)?,
        query_step: p["query_step"].as_f64().ok_or("query step")?,
        learning_rate: p["learning_rate"].as_f64().ok_or("learning rate")?,
        line_iterations: p["line_iterations"].as_u64().ok_or("line iterations")? as usize,
    };
    Ok((p, config))
}
pub fn train(
    workspace: &Path,
    storage: &Path,
    sources: &Path,
    cohort: &Path,
    variant: &str,
    mode: &str,
    text: &str,
) -> Result<()> {
    guard::admit(0, workspace, storage)?;
    let (protocol, config) = settings_with(variant, mode, text)?;
    let (mut model, row, input, task) = source::load(workspace, storage, sources, variant, mode)?;
    let batch = row["case"]["batch"].as_u64().ok_or("original batch")? as usize;
    let updates = protocol["updates"].as_u64().ok_or("updates")? as usize;
    let reference = continuation::Replay::load(storage, &protocol, variant, mode, &input)?;
    let path = cell(cohort, variant, mode);
    fs::create_dir_all(path.parent().ok_or("parent")?)?;
    fs::create_dir(&path)?;
    original::write_json(
        &path.join("metadata.json"),
        &json!({"source_fingerprint":fingerprint(),
        "protocol":protocol,"input":input,"original":row,"objects":model.objects,"budget":model.budget,
        "prefix_reference":reference.as_ref().map(|r|&r.identity),
        "variant":variant,"mode":mode,"selection":task.selection,
        "binary_sha256":original::hash(&fs::read(std::env::current_exe()?)?)}),
    )?;
    let mut trace = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path.join("training.jsonl"))?;
    let training = &task.splits["train"];
    let mut order: Vec<_> = (0..training.len()).collect();
    let mut rng = Random::new(10_001);
    let (mut cursor, mut examples, mut targets, mut queries) =
        (order.len(), 0usize, 0usize, 0usize);
    let mut bias_queries = 0usize;
    let mut line_queries = 0usize;
    let start = Instant::now();
    for update in 0..updates {
        if cursor == order.len() {
            rng.shuffle(&mut order);
            cursor = 0;
        }
        let end = (cursor + batch).min(order.len());
        let records = original::records(training, &order[cursor..end])?;
        let rss = guard::admit(
            model
                .reservation(records.len())?
                .checked_mul(4)
                .ok_or("refit reservation")?,
            workspace,
            storage,
        )?;
        let receipt = update::step(&mut model, &records, config)?;
        examples += end - cursor;
        targets += records.len();
        queries += receipt.total_ce_evaluations;
        bias_queries += receipt.bias_loss_queries;
        line_queries += receipt.line_loss_queries;
        let indices: Vec<_> = order[cursor..end]
            .iter()
            .flat_map(|&i| (i as u64).to_le_bytes())
            .collect();
        cursor = end;
        let record = json!({"update":update+1,"examples_seen":examples,"targets_seen":targets,
            "batch_indices_sha256":original::hash(&indices),"aggregate_rss_before":rss,
            "elapsed_seconds":start.elapsed().as_secs_f64(),"receipt":receipt});
        if let Some(reference) = &reference {
            reference.observe(update, &record, &model, &path)?;
        }
        writeln!(trace, "{record}")?;
        trace.flush()?;
        if update == 0 || (update + 1) % 64 == 0 {
            println!(
                "source={variant} mode={mode} update={} centre={:.8} after={:.8} eta={:.6}",
                update + 1,
                receipt.centre_ce,
                receipt.after_ce,
                receipt.eta
            );
        }
    }
    let content = source::content_hash(&model)?;
    if input["content_sha256"] != content {
        return Err("conditional refit changed learned content".into());
    }
    original::write_json(&path.join("endpoint.json"), &model)?;
    original::write_json(
        &path.join("training_receipt.json"),
        &json!({"source_fingerprint":fingerprint(),
        "variant":variant,"mode":mode,"updates":updates,"examples_seen":examples,"targets_seen":targets,
        "epochs":examples as f64/training.len() as f64,"total_ce_evaluations":queries,
        "bias_loss_queries":bias_queries,"line_loss_queries":line_queries,
        "centre_validation_and_check_ce_evaluations":3*targets,"components_stored":model.atoms.len(),
        "content_sha256":content,"elapsed_seconds":start.elapsed().as_secs_f64(),
        "endpoint_sha256":original::hash(&fs::read(path.join("endpoint.json"))?),
        "status":"sealed","heldout_scored":false}),
    )?;
    Ok(())
}

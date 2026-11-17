//! Fixed prospective minibatch schedule. All four endpoints seal before scores.
use crate::{Result, experiment, update};
use baseline::random::Random;
use bilinear::{experiment as original, guard};
use serde_json::json;
use std::{fs, io::Write, path::Path, time::Instant};

pub fn run(
    workspace: &Path,
    storage: &Path,
    sources: &Path,
    cohort: &Path,
    variant: &str,
    mode: &str,
) -> Result<()> {
    guard::admit(0, workspace, storage)?;
    let (p, config) = experiment::settings(variant, mode)?;
    let (mut model, input, task) = optimality::source::load(
        workspace,
        storage,
        sources,
        p["training_input_cohort"].as_str().ok_or("input cohort")?,
        variant,
        mode,
    )?;
    let (_, row, _) = original::case_in(
        readouts::source::CASE,
        rotary::experiment::CAUTIOUS_PROTOCOL,
    )?;
    let batch = p["batch"].as_u64().ok_or("batch")? as usize;
    let updates = p["updates"].as_u64().ok_or("updates")? as usize;
    if row["case"]["batch"] != batch {
        return Err("original batch size changed".into());
    }
    let path = experiment::cell(cohort, variant, mode);
    fs::create_dir_all(path.parent().ok_or("parent")?)?;
    fs::create_dir(&path)?;
    original::write_json(
        &path.join("metadata.json"),
        &json!({
        "source_fingerprint":experiment::fingerprint(),"protocol":p,"input":input,
        "original":row,"objects":model.objects,"budget":model.budget,
        "variant":variant,"mode":mode,"selection":task.selection,
        "binary_sha256":original::hash(&fs::read(std::env::current_exe()?)?)}),
    )?;
    let mut trace = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path.join("training.jsonl"))?;
    let training = &task.splits["train"];
    let mut order: Vec<_> = (0..training.len()).collect();
    let mut rng = Random::new(
        p["shuffle_seed"]
            .as_u64()
            .ok_or("shuffle seed")?
            .try_into()?,
    );
    let (mut cursor, mut examples, mut targets, mut queries, mut accepted) =
        (order.len(), 0usize, 0usize, 0usize, 0usize);
    let (mut bias, mut line) = (0usize, 0usize);
    let mut chosen = vec![0usize; model.atoms.len()];
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
                .checked_mul(6)
                .ok_or("training reservation")?,
            workspace,
            storage,
        )?;
        let receipt = update::step(&mut model, &records, config)?;
        examples += end - cursor;
        targets += records.len();
        queries += receipt.total_ce_evaluations;
        bias += receipt.bias_loss_queries;
        line += receipt.line_loss_queries;
        accepted += usize::from(receipt.eta > 0.);
        chosen[receipt.selected_component] += 1;
        let ids: Vec<_> = order[cursor..end]
            .iter()
            .flat_map(|&i| (i as u64).to_le_bytes())
            .collect();
        cursor = end;
        writeln!(
            trace,
            "{}",
            json!({"update":update+1,"examples_seen":examples,"targets_seen":targets,
            "batch_indices_sha256":original::hash(&ids),"aggregate_rss_before":rss,
            "elapsed_seconds":start.elapsed().as_secs_f64(),"receipt":receipt})
        )?;
        trace.flush()?;
        if update == 0 || (update + 1) % 64 == 0 {
            println!(
                "{variant}/{mode} update={} centre={:.8} after={:.8} eta={:.6} selected={}",
                update + 1,
                receipt.centre_ce,
                receipt.after_ce,
                receipt.eta,
                receipt.selected_component
            );
        }
    }
    let content = readouts::source::content_hash(&model)?;
    if input["content_sha256"] != content {
        return Err("joint phase changed learned content".into());
    }
    original::write_json(&path.join("endpoint.json"), &model)?;
    original::write_json(
        &path.join("training_receipt.json"),
        &json!({
        "source_fingerprint":experiment::fingerprint(),"variant":variant,"mode":mode,
        "updates":updates,"accepted_updates":accepted,"examples_seen":examples,"targets_seen":targets,
        "epochs":examples as f64/training.len() as f64,"total_ce_evaluations":queries,
        "bias_loss_queries":bias,"line_loss_queries":line,"centre_and_check_ce_evaluations":3*targets,
        "components_stored":model.atoms.len(),"selected_component_counts":chosen,"content_sha256":content,
        "elapsed_seconds":start.elapsed().as_secs_f64(),"endpoint_sha256":original::hash(&fs::read(path.join("endpoint.json"))?),
        "status":"sealed","heldout_scored":false}),
    )?;
    Ok(())
}

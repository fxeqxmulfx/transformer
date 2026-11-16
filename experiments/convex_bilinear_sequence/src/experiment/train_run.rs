//! Train on original indexed examples; seal the endpoint before held-out scoring.
use super::*;
use crate::{
    algebra, guard,
    model::{Joint, Model},
    train,
};
use serde_json::json;
use std::{collections::BTreeSet, fs, io::Write, path::Path, time::Instant};

pub fn run(
    workspace: &Path,
    storage: &Path,
    name: &str,
    mode_name: &str,
    seed: u64,
    output: &Path,
    protocol_text: &str,
) -> Result<()> {
    let started = Instant::now();
    let protocol: Value = serde_json::from_str(protocol_text)?;
    if !protocol["seeds"]
        .as_array()
        .ok_or("missing seeds")?
        .contains(&json!(seed))
    {
        return Err("seed outside prospective protocol".into());
    }
    let (config, original, task) = case_in(name, protocol_text)?;
    let shape = shape(&task)?;
    let mode = mode(mode_name)?;
    let max_bytes = protocol["model_reservation_bytes"]
        .as_u64()
        .ok_or("memory budget")?;
    guard::admit(shape.reservation(0)?, workspace, storage)?;
    algebra::set_threads(protocol["blas_threads"].as_i64().ok_or("threads")? as i32)?;
    let mut model = Model::recover(Joint::zero(shape), max_bytes)?;
    let training = &task.splits["train"];
    let scaling = match protocol["diagonal_scaling"].as_str() {
        None => None,
        Some("training-response-frequency") => {
            let records = records(training, &(0..training.len()).collect::<Vec<_>>())?;
            let mut columns = BTreeSet::new();
            for record in &records {
                columns.extend(
                    shape
                        .addresses(record.prefix)?
                        .columns
                        .into_iter()
                        .map(|(d, _)| d),
                );
            }
            guard::admit(shape.reservation(columns.len())?, workspace, storage)?;
            Some(crate::scaling::FrequencyScale::from_prefixes(
                shape,
                records.iter().map(|r| r.prefix),
            )?)
        }
        Some(_) => return Err("unknown diagonal scaling".into()),
    };
    let batch_size = original["case"]["batch"].as_u64().ok_or("original batch")? as usize;
    let updates = config["maximum_updates"].as_u64().ok_or("update budget")? as usize;
    let rate = protocol["learning_rate"].as_f64().ok_or("learning rate")?;
    let query_step = protocol["forward_query_step"]
        .as_f64()
        .ok_or("query step")?;
    fs::create_dir_all(output.parent().ok_or("missing output parent")?)?;
    fs::create_dir(output)?;
    write_json(
        &output.join("metadata.json"),
        &json!({"source_fingerprint":fingerprint(), "protocol":protocol,
        "case":config,"original":original,"mode":mode_name,"seed":seed,"shape":shape,
        "complete_real_coordinates":shape.real_coordinates().to_string(),"selection":task.selection,
        "binary_sha256":hash(&fs::read(std::env::current_exe()?)?)}),
    )?;
    let mut trace = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output.join("training.jsonl"))?;
    let mut rng = baseline::random::Random::new(u32::try_from(
        seed.checked_add(10_000).ok_or("seed overflow")?,
    )?);
    let mut order: Vec<_> = (0..training.len()).collect();
    let (mut cursor, mut examples_seen, mut targets_seen, mut queries) =
        (training.len(), 0usize, 0usize, 0usize);
    let mut max_residual = 0f64;
    for update in 0..updates {
        if cursor == order.len() {
            rng.shuffle(&mut order);
            cursor = 0;
        }
        let end = (cursor + batch_size).min(order.len());
        let records = records(training, &order[cursor..end])?;
        let mut active: BTreeSet<_> = model.joint.columns.keys().copied().collect();
        for record in &records {
            active.extend(
                shape
                    .addresses(record.prefix)?
                    .columns
                    .into_iter()
                    .map(|(d, _)| d),
            );
        }
        let rss = guard::admit(shape.reservation(active.len())?, workspace, storage)?;
        let (next, receipt) =
            train::step_scaled(&model, &records, mode, rate, query_step, scaling.as_ref())?;
        examples_seen += end - cursor;
        targets_seen += records.len();
        queries += receipt.bias_loss_queries;
        max_residual = max_residual.max(receipt.maximum_forward_residual);
        model = next;
        cursor = end;
        writeln!(
            trace,
            "{}",
            json!({"update":update + 1,"examples_seen":examples_seen,"targets_seen":targets_seen,
            "elapsed_seconds":started.elapsed().as_secs_f64(),"aggregate_rss_before":rss,"receipt":receipt})
        )?;
        trace.flush()?;
        if update == 0 || (update + 1) % 25 == 0 {
            println!(
                "case={name} mode={mode_name} update={} records={} ce={:.8} columns={} seconds={:.2}",
                update + 1,
                targets_seen,
                receipt.mean_ce_before,
                model.joint.columns.len(),
                started.elapsed().as_secs_f64()
            );
        }
    }
    write_json(&output.join("endpoint.json"), &model.joint)?;
    let endpoint_hash = hash(&fs::read(output.join("endpoint.json"))?);
    write_json(
        &output.join("training_receipt.json"),
        &json!({"source_fingerprint":fingerprint(),"case":name,
        "mode":mode_name,"seed":seed,"updates":updates,"examples_seen":examples_seen,"targets_seen":targets_seen,
        "epochs":examples_seen as f64 / training.len() as f64,"bias_loss_queries":queries,
        "elapsed_seconds":started.elapsed().as_secs_f64(),"endpoint_sha256":endpoint_hash,
        "maximum_forward_residual":max_residual,"reconstruction_error":model.reconstruction_error,
        "balance_error":model.balance_error,"stored_columns":model.joint.columns.len(),
        "status":"sealed","heldout_scored":false}),
    )?;
    Ok(())
}

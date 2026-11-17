//! Twelve endpoints, all sealed before any held-out scoring; fixed original sampler.
use crate::{Result, cache::Cache, experiment, index::Index, update};
use baseline::random::Random;
use bilinear::{experiment as original, guard};
use serde_json::json;
use std::{fs, io::Write, path::Path, time::Instant};

pub struct Paths<'a> {
    pub workspace: &'a Path,
    pub storage: &'a Path,
    pub sources: &'a Path,
    pub cohort: &'a Path,
}
pub fn run(paths: Paths<'_>, arm: &str, variant: &str, mode: &str) -> Result<()> {
    let Paths {
        workspace,
        storage,
        sources,
        cohort,
    } = paths;
    guard::admit(0, workspace, storage)?;
    let (p, config, retained) = experiment::settings(arm, variant, mode)?;
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
    let training = &task.splits["train"];
    let index = Index::new(training)?;
    let mut order: Vec<_> = (0..training.len()).collect();
    let (_, all_records) = index.batch(training, &order)?;
    guard::admit(
        model
            .reservation(all_records.len())?
            .checked_mul(6)
            .ok_or("cache reservation")?,
        workspace,
        storage,
    )?;
    let mut cache = if retained {
        Some(Cache::new(&model, &all_records)?)
    } else {
        None
    };
    drop(all_records);
    let path = experiment::cell(cohort, arm, variant, mode);
    fs::create_dir_all(path.parent().ok_or("parent")?)?;
    fs::create_dir(&path)?;
    original::write_json(
        &path.join("metadata.json"),
        &json!({"source_fingerprint":experiment::fingerprint(),
        "protocol":p,"input":input,"original":row,"objects":model.objects,"budget":model.budget,
        "arm":arm,"variant":variant,"mode":mode,"selection":task.selection,
        "binary_sha256":original::hash(&fs::read(std::env::current_exe()?)?)}),
    )?;
    let mut trace = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path.join("training.jsonl"))?;
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
    for n in 0..updates {
        if cursor == order.len() {
            rng.shuffle(&mut order);
            cursor = 0;
        }
        let end = (cursor + batch).min(order.len());
        let (ids, records) = index.batch(training, &order[cursor..end])?;
        let rss = guard::admit(
            model
                .reservation(records.len())?
                .checked_mul(6)
                .ok_or("update reservation")?,
            workspace,
            storage,
        )?;
        let receipt = if let Some(c) = &mut cache {
            update::step(&mut model, c, &ids, &records, config)?
        } else {
            joint::update::step(&mut model, &records, config)?
        };
        examples += end - cursor;
        targets += records.len();
        queries += receipt.total_ce_evaluations;
        bias += receipt.bias_loss_queries;
        line += receipt.line_loss_queries;
        accepted += usize::from(receipt.eta > 0.);
        chosen[receipt.selected_component] += 1;
        let example_bytes: Vec<_> = order[cursor..end]
            .iter()
            .flat_map(|&i| (i as u64).to_le_bytes())
            .collect();
        let record_bytes: Vec<_> = ids.iter().flat_map(|&i| (i as u64).to_le_bytes()).collect();
        cursor = end;
        writeln!(
            trace,
            "{}",
            json!({"update":n+1,"examples_seen":examples,"targets_seen":targets,
            "batch_indices_sha256":original::hash(&example_bytes),"record_indices_sha256":original::hash(&record_bytes),
            "cache_coverage":cache.as_ref().map(Cache::coverage),"aggregate_rss_before":rss,
            "elapsed_seconds":start.elapsed().as_secs_f64(),"receipt":receipt})
        )?;
        trace.flush()?;
        if n == 0 || (n + 1) % 64 == 0 {
            println!(
                "{arm}/{variant}/{mode} update={} centre={:.8} after={:.8} eta={:.6} selected={}",
                n + 1,
                receipt.centre_ce,
                receipt.after_ce,
                receipt.eta,
                receipt.selected_component
            );
        }
    }
    let content = readouts::source::content_hash(&model)?;
    if input["content_sha256"] != content {
        return Err("retained phase changed learned content".into());
    }
    let cache_hash = if let Some(c) = &cache {
        c.validate(&model)?;
        original::write_json(&path.join("cache.json"), c)?;
        Some(original::hash(&fs::read(path.join("cache.json"))?))
    } else {
        None
    };
    original::write_json(&path.join("endpoint.json"), &model)?;
    original::write_json(
        &path.join("training_receipt.json"),
        &json!({"source_fingerprint":experiment::fingerprint(),
        "arm":arm,"variant":variant,"mode":mode,"updates":updates,"accepted_updates":accepted,
        "examples_seen":examples,"targets_seen":targets,"epochs":examples as f64/training.len() as f64,
        "total_ce_evaluations":queries,"bias_loss_queries":bias,"line_loss_queries":line,
        "centre_and_check_ce_evaluations":3*targets,"components_stored":model.atoms.len(),
        "selected_component_counts":chosen,"content_sha256":content,"cache_sha256":cache_hash,
        "cache_coverage":cache.as_ref().map(Cache::coverage),"cache_refreshes":cache.as_ref().map(|c|c.refreshed),
        "elapsed_seconds":start.elapsed().as_secs_f64(),"endpoint_sha256":original::hash(&fs::read(path.join("endpoint.json"))?),
        "status":"sealed","heldout_scored":false}),
    )?;
    Ok(())
}

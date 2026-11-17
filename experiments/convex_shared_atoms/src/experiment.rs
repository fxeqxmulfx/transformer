//! Frozen original data and protocol. No task identity enters model inference.
use crate::{
    Result,
    model::Model,
    oracle::Gradient,
    train::{self, Config},
};
use baseline::random::Random;
use bilinear::{experiment as original, guard};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, io::Write, path::Path, time::Instant};

pub mod score;
pub const PROTOCOL: &str = include_str!("../protocol.json");
pub const BALANCED_PROTOCOL: &str = include_str!("../balanced_protocol.json");
pub const THREE_SEED_PROTOCOL: &str = include_str!("../three_seed_protocol.json");
pub const REFINED_PROTOCOL: &str = include_str!("../refined_protocol.json");

pub fn fingerprint() -> String {
    let mut digest = Sha256::new();
    for bytes in [
        include_bytes!("lib.rs").as_slice(),
        include_bytes!("model.rs"),
        include_bytes!("oracle.rs"),
        include_bytes!("train.rs"),
        include_bytes!("experiment.rs"),
        include_bytes!("experiment/score.rs"),
        include_bytes!("bin/stress.rs"),
        include_bytes!("../Cargo.toml"),
        include_bytes!("../Cargo.lock"),
        PROTOCOL.as_bytes(),
        BALANCED_PROTOCOL.as_bytes(),
        THREE_SEED_PROTOCOL.as_bytes(),
        REFINED_PROTOCOL.as_bytes(),
        original::fingerprint().as_bytes(),
    ] {
        digest.update((bytes.len() as u64).to_le_bytes());
        digest.update(bytes);
    }
    digest
        .finalize()
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}

pub fn protocol(mode: &str, seed: u64, text: &str) -> Result<(Value, Config)> {
    let p: Value = serde_json::from_str(text)?;
    if !p["modes"].as_array().ok_or("modes")?.contains(&json!(mode))
        || !p["seeds"].as_array().ok_or("seeds")?.contains(&json!(seed))
    {
        return Err("mode/seed outside frozen cohort".into());
    }
    let config = Config {
        mode: original::mode(mode)?,
        query_step: p["query_step"].as_f64().ok_or("query step")?,
        momentum: p["momentum"].as_f64().ok_or("momentum")?,
        restarts: p["component_restarts"].as_u64().ok_or("restarts")? as usize,
        sweeps: p["component_sweeps"].as_u64().ok_or("sweeps")? as usize,
        line_iterations: p["line_iterations"].as_u64().ok_or("line iterations")? as usize,
    };
    Ok((p, config))
}

pub fn initial_model(shape: bilinear::domain::Shape, protocol: &Value) -> Result<Model> {
    let budget = protocol["budget_multiplier"].as_f64().ok_or("capacity")?
        * ((shape.objects as f64) * (shape.queries as f64) * (shape.columns as f64)).sqrt();
    match protocol["geometry"].as_str() {
        None | Some("identity") => Model::zero(shape, budget),
        Some("dimension-bias") => Model::balanced(shape, budget),
        _ => Err("unknown positive coordinate geometry".into()),
    }
}

pub fn train(
    workspace: &Path,
    storage: &Path,
    cohort: &Path,
    name: &str,
    mode: &str,
    seed: u64,
    text: &str,
) -> Result<()> {
    guard::admit(0, workspace, storage)?;
    let (protocol, config) = protocol(mode, seed, text)?;
    let (case, original_row, task) = original::case_in(name, text)?;
    let shape = original::shape(&task)?;
    let mut model = initial_model(shape, &protocol)?;
    let budget = model.budget;
    let training = &task.splits["train"];
    let batch = original_row["case"]["batch"]
        .as_u64()
        .ok_or("original batch")? as usize;
    let updates = case["maximum_updates"].as_u64().ok_or("updates")? as usize;
    guard::admit(model.reservation(batch)?, workspace, storage)?;
    let mut gradient = Gradient::zero(shape)?;
    let path = original::score::cell(cohort, name, mode, seed);
    fs::create_dir_all(path.parent().ok_or("output parent")?)?;
    fs::create_dir(&path)?;
    original::write_json(
        &path.join("metadata.json"),
        &json!({"source_fingerprint":fingerprint(),
        "protocol":protocol,"case":case,"original":original_row,"shape":shape,"selection":task.selection,
        "seed":seed,"mode":mode,"budget":budget,"complete_real_coordinates":shape.real_coordinates().to_string(),
        "query_scale":model.query_scale,"context_scale":model.context_scale,
        "raw_bias_response":model.query_scale[0]*model.context_scale[0],
        "binary_sha256":original::hash(&fs::read(std::env::current_exe()?)?)}),
    )?;
    let mut trace = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path.join("training.jsonl"))?;
    let mut sampler = Random::new(u32::try_from(
        seed.checked_add(10_000).ok_or("seed overflow")?,
    )?);
    let mut component_rng = Random::new(u32::try_from(
        seed.checked_add(20_000).ok_or("seed overflow")?,
    )?);
    let mut order: Vec<_> = (0..training.len()).collect();
    let (mut cursor, mut examples, mut targets, mut queries) =
        (order.len(), 0usize, 0usize, 0usize);
    let started = Instant::now();
    for update in 0..updates {
        if cursor == order.len() {
            sampler.shuffle(&mut order);
            cursor = 0;
        }
        let end = (cursor + batch).min(order.len());
        let records = original::records(training, &order[cursor..end])?;
        let reservation = model.reservation(records.len())?;
        if reservation
            > protocol["model_reservation_bytes"]
                .as_u64()
                .ok_or("model budget")?
        {
            return Err("complete model allocation exceeds reservation cap".into());
        }
        let rss = guard::admit(reservation, workspace, storage)?;
        let receipt = train::step(
            &mut model,
            &mut gradient,
            &records,
            config,
            &mut component_rng,
        )?;
        examples += end - cursor;
        targets += records.len();
        queries += receipt.bias_loss_queries;
        let indices: Vec<_> = order[cursor..end]
            .iter()
            .flat_map(|&i| (i as u64).to_le_bytes())
            .collect();
        cursor = end;
        writeln!(
            trace,
            "{}",
            json!({"update":update+1,"examples_seen":examples,"targets_seen":targets,
            "batch_indices_sha256":original::hash(&indices),"aggregate_rss_before":rss,
            "elapsed_seconds":started.elapsed().as_secs_f64(),"receipt":receipt})
        )?;
        trace.flush()?;
        if update == 0 || (update + 1) % 25 == 0 {
            println!(
                "case={name} mode={mode} update={} ce={:.8} after={:.8} eta={:.6} atoms={} seconds={:.2}",
                update + 1,
                receipt.centre_ce,
                receipt.after_ce,
                receipt.eta,
                model.atoms.len(),
                started.elapsed().as_secs_f64()
            );
        }
    }
    original::write_json(&path.join("endpoint.json"), &model)?;
    original::write_json(
        &path.join("training_receipt.json"),
        &json!({"source_fingerprint":fingerprint(),
        "case":name,"mode":mode,"seed":seed,"updates":updates,"examples_seen":examples,"targets_seen":targets,
        "epochs":examples as f64/training.len() as f64,"bias_loss_queries":queries,
        "components_stored":model.atoms.len(),"elapsed_seconds":started.elapsed().as_secs_f64(),
        "endpoint_sha256":original::hash(&fs::read(path.join("endpoint.json"))?),
        "status":"sealed","heldout_scored":false}),
    )?;
    Ok(())
}

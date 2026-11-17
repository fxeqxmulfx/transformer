//! Frozen original data and protocol. No task identity enters model inference.
//! Cohort machinery ported from scalar recurrence1786526f; matrix/rotary
//! capacities replace scalar geometry. Training never reads held-out targets.
use crate::{
    Result,
    atom::Family,
    model::Model,
    train::{self, Config},
};
use baseline::random::Random;
use bilinear::{experiment as original, guard};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, io::Write, path::Path, time::Instant};

pub mod score;
pub const PROTOCOL: &str = include_str!("../protocol.json");
pub const CAUTIOUS_PROTOCOL: &str = include_str!("../cautious_protocol.json");
pub const REFINED_PROTOCOL: &str = include_str!("../refined_protocol.json");

pub fn fingerprint() -> String {
    let mut digest = Sha256::new();
    for bytes in [
        include_bytes!("lib.rs").as_slice(),
        include_bytes!("model.rs"),
        include_bytes!("atom.rs"),
        include_bytes!("coordinates.rs"),
        include_bytes!("proposal.rs"),
        include_bytes!("proposal/refinement.rs"),
        include_bytes!("train.rs"),
        include_bytes!("experiment.rs"),
        include_bytes!("experiment/score.rs"),
        include_bytes!("bin/stress.rs"),
        include_bytes!("../Cargo.toml"),
        include_bytes!("../Cargo.lock"),
        PROTOCOL.as_bytes(),
        CAUTIOUS_PROTOCOL.as_bytes(),
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
        sweeps: p["component_sweeps"].as_u64().ok_or("sweeps")? as usize,
        line_iterations: p["line_iterations"].as_u64().ok_or("line iterations")? as usize,
        step_ceiling: step_ceiling(&p, 0)?,
        frequency_start: match p["frequency_initialization"].as_str() {
            None => None,
            Some("matched-fixed")
                if p["fixed_frequency"] == json!(Family::Fixed.initial_frequency()) =>
            {
                Family::Fixed.initial_frequency()
            }
            _ => return Err("unknown frequency initialization".into()),
        },
        frequency_refinement_steps: match p.get("frequency_refinement_steps") {
            None => 0,
            Some(v) => usize::try_from(v.as_u64().ok_or("frequency refinement steps")?)?,
        },
    };
    Ok((p, config))
}

/// CM99 ceiling at the zero-based update; a feasibility condition, not convergence.
pub fn step_ceiling(protocol: &Value, update: usize) -> Result<f64> {
    match protocol["step_schedule"].as_str() {
        None => Ok(1.),
        Some("two-over-t-plus-two") => Ok(2. / (update as f64 + 2.)),
        _ => Err("unknown convex step schedule".into()),
    }
}

pub fn initial_model(objects: usize, protocol: &Value, family: Family) -> Result<Model> {
    if family == Family::Fixed && protocol["fixed_frequency"] != json!(family.initial_frequency()) {
        return Err("fixed frequency differs from declared model".into());
    }
    if !protocol["families"]
        .as_array()
        .ok_or("families")?
        .contains(&json!(family))
    {
        return Err("family outside frozen comparison".into());
    }
    Model::zero(
        objects,
        protocol["budget"].as_f64().ok_or("capacity")?,
        family,
    )
}

pub fn train(
    workspace: &Path,
    storage: &Path,
    cohort: &Path,
    name: &str,
    mode: &str,
    seed: u64,
    options: (&str, Family),
) -> Result<()> {
    let (text, family) = options;
    guard::admit(0, workspace, storage)?;
    let (protocol, config) = protocol(mode, seed, text)?;
    let (case, original_row, task) = original::case_in(name, text)?;
    let mut model = initial_model(task.vocab, &protocol, family)?;
    let budget = model.budget;
    let training = &task.splits["train"];
    let batch = original_row["case"]["batch"]
        .as_u64()
        .ok_or("original batch")? as usize;
    let updates = case["maximum_updates"].as_u64().ok_or("updates")? as usize;
    guard::admit(model.reservation(batch)?, workspace, storage)?;
    let path = score::cell(cohort, name, mode, seed, family);
    fs::create_dir_all(path.parent().ok_or("output parent")?)?;
    fs::create_dir(&path)?;
    original::write_json(
        &path.join("metadata.json"),
        &json!({"source_fingerprint":fingerprint(),
        "protocol":protocol,"case":case,"original":original_row,"objects":task.vocab,"selection":task.selection,
        "seed":seed,"mode":mode,"family":family,"budget":budget,"real_coordinates_per_component":family.coordinates(task.vocab),
        "trainable_real_coordinates_per_component":family.trainable_coordinates(task.vocab),
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
    let (mut component_queries, mut line_queries) = (0usize, 0usize);
    let mut coordinate_trials = [0usize; 3];
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
        let config = Config {
            step_ceiling: step_ceiling(&protocol, update)?,
            ..config
        };
        let receipt = train::step(&mut model, &records, config, &mut component_rng)?;
        examples += end - cursor;
        targets += records.len();
        queries += receipt.bias_loss_queries;
        component_queries += receipt.component_loss_queries;
        for (total, count) in coordinate_trials.iter_mut().zip(receipt.coordinate_trials) {
            *total += count;
        }
        line_queries += receipt.line_loss_queries;
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
                "case={name} family={family:?} mode={mode} update={} ce={:.8} after={:.8} eta={:.6} atoms={} seconds={:.2}",
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
        "case":name,"mode":mode,"family":family,"seed":seed,"updates":updates,"examples_seen":examples,"targets_seen":targets,
        "epochs":examples as f64/training.len() as f64,"bias_loss_queries":queries,
        "component_loss_queries":component_queries,"coordinate_trials":coordinate_trials,"line_loss_queries":line_queries,
        "centre_and_validation_ce_evaluations":2*targets,
        "total_ce_evaluations":queries+component_queries+line_queries+2*targets,
        "components_stored":model.atoms.len(),"elapsed_seconds":started.elapsed().as_secs_f64(),
        "endpoint_sha256":original::hash(&fs::read(path.join("endpoint.json"))?),
        "status":"sealed","heldout_scored":false}),
    )?;
    Ok(())
}

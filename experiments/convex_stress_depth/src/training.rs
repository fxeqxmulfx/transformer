//! Training-only checks. No validation/test metric participates or is computed.
//! Every mode pays for the same empirical forward initializer before optional
//! gradient refinement. These are not from-zero optimizer comparisons.
use crate::{
    Result, checkpoint, data,
    features::{Bank, Transport},
    head::{self, Cache, Head},
};
use convex_stress_hull::{meta, random::Random};
use serde_json::{Value, json};
use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Write},
    path::Path,
    time::Instant,
};

pub const PROTOCOL: &str = include_str!("../train_check.json");
pub fn fingerprint() -> String {
    let mut parts = vec![];
    let sources: &[&[u8]] = &[
        include_bytes!("lib.rs"),
        include_bytes!("main.rs"),
        include_bytes!("features.rs"),
        include_bytes!("head.rs"),
        include_bytes!("independent.rs"),
        include_bytes!("checkpoint.rs"),
        include_bytes!("training.rs"),
        include_bytes!("stream_cache.rs"),
        include_bytes!("stream_cache/objective.rs"),
        include_bytes!("../Cargo.toml"),
        include_bytes!("../Cargo.lock"),
        include_bytes!("../run_train_checks.sh"),
        PROTOCOL.as_bytes(),
    ];
    for source in sources {
        parts.extend_from_slice(&(source.len() as u64).to_le_bytes());
        parts.extend_from_slice(source);
    }
    parts.extend_from_slice(meta::fingerprint().as_bytes());
    crate::hash(&parts)
}
fn write_json(path: impl AsRef<Path>, value: &Value) -> Result<()> {
    let mut out = BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(&mut out, value)?;
    out.write_all(b"\n")?;
    out.flush()?;
    Ok(())
}
pub fn run(task_name: &str, depth: usize, mode: &str, directory: &Path) -> Result<()> {
    let start = Instant::now();
    let protocol: Value = serde_json::from_str(PROTOCOL)?;
    if !meta::allowed(&protocol, "tasks", task_name)
        || !meta::allowed(&protocol, "modes", mode)
        || !protocol["depths"]
            .as_array()
            .unwrap()
            .iter()
            .any(|d| d.as_u64() == Some(depth as u64))
    {
        return Err("unsupported training-only check".into());
    }
    if protocol["initializer_log_frequency_floor"] != -24 || protocol["heldout_scoring"] != false {
        return Err("training-only protocol disagrees with implementation".into());
    }
    let commit = std::env::var("DEPTH_PRODUCER_COMMIT")?;
    if commit.len() != 40 || !commit.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err("DEPTH_PRODUCER_COMMIT must identify the committed producer".into());
    }
    fs::create_dir(directory)?;
    let row = meta::selected(task_name)?;
    let task = data::build(&row)?;
    let data_seconds = start.elapsed().as_secs_f64();
    let train = &task.splits["train"];
    let bank = Bank::discover(
        task.vocab,
        depth,
        protocol["history_slots"].as_u64().unwrap() as usize,
        Transport::Fourier,
        train,
    )?;
    let n: usize = train
        .iter()
        .map(|e| e.targets.iter().filter(|t| **t >= 0).count())
        .sum();
    let allocation = bank.allocation_bytes(n)?;
    let metadata = json!({"status":"started","source_fingerprint":fingerprint(), "producer_commit":commit,
        "task":task_name,"depth":depth,"mode":mode,"protocol":protocol,"data_fingerprints":row["data_fingerprints"],
        "training_examples":train.len(),"training_prefixes":n,"bank":bank,"explicit_buffer_bytes":allocation,
        "heldout_scored":false});
    write_json(directory.join("metadata.json"), &metadata)?;
    if allocation > head::BUFFER_LIMIT {
        write_json(
            directory.join("result.json"),
            &json!({"status":"ineligible_memory_budget", "heldout_scored":false,
            "explicit_buffer_bytes":allocation,"source_fingerprint":fingerprint()}),
        )?;
        return Ok(());
    }
    let encoded_at = Instant::now();
    let cache = Cache::build(&bank, train)?;
    let feature_seconds = encoded_at.elapsed().as_secs_f64();
    let mut model = Head::zero(
        bank,
        protocol["radius_times_vocab"].as_f64().unwrap() * task.vocab as f64,
    )?;
    let mut gradient = vec![0.; model.coefficients.len()];
    let init_at = Instant::now();
    let initializer_channel = head::forward_initialize(&mut model, &cache, &mut gradient)?;
    let initializer_seconds = init_at.elapsed().as_secs_f64();
    let order = cache.full_order();
    let initial_ce = head::objective(&model, &cache, &order, None)?.0;
    let mut trace = BufWriter::new(File::create(directory.join("trace.jsonl"))?);
    writeln!(
        trace,
        "{}",
        json!({"update":0,"train_ce":initial_ce,"elapsed_seconds":start.elapsed().as_secs_f64()})
    )?;
    trace.flush()?;
    let (mut updates, mut examples_seen, mut prefixes_seen) = (0usize, 0usize, 0usize);
    let mut ce = initial_ce;
    let mut shuffled = order.clone();
    let mut rng = Random::new(1);
    let mut cursor = shuffled.len();
    let batch_size = row["case"]["batch"]
        .as_u64()
        .ok_or("missing original batch size")? as usize;
    let cap = match mode {
        "full_batch" => protocol["full_batch_updates"].as_u64().unwrap() as usize,
        "minibatch" => protocol["minibatch_updates"].as_u64().unwrap() as usize,
        _ => 0,
    };
    let optimization_at = Instant::now();
    for step in 1..=cap {
        let batch = if mode == "full_batch" {
            order.as_slice()
        } else {
            if cursor == shuffled.len() {
                rng.shuffle(&mut shuffled);
                cursor = 0;
            }
            let end = (cursor + batch_size).min(shuffled.len());
            let begin = cursor;
            cursor = end;
            &shuffled[begin..end]
        };
        let (_, count) = head::objective(&model, &cache, batch, Some(&mut gradient))?;
        head::gradient_step(&mut model, &gradient, 1. / cache.width as f64)?;
        updates = step;
        examples_seen += batch.len();
        prefixes_seen += count;
        if mode == "full_batch" || [1, 16, 64, 256].contains(&step) || step == cap {
            ce = head::objective(&model, &cache, &order, None)?.0;
            writeln!(
                trace,
                "{}",
                json!({"update":step,"train_ce":ce,"gradient_examples_seen":examples_seen,
                "gradient_prefixes_seen":prefixes_seen,"elapsed_seconds":start.elapsed().as_secs_f64()})
            )?;
            trace.flush()?;
            if step >= protocol["minimum_gradient_updates"].as_u64().unwrap() as usize
                && ce <= protocol["training_ce_stop"].as_f64().unwrap()
            {
                break;
            }
        }
    }
    let optimization_seconds = optimization_at.elapsed().as_secs_f64();
    let audit_at = Instant::now();
    // Forward-only runs never form a loss gradient, including their final check.
    let gap = if mode == "forward_only" {
        None
    } else {
        ce = head::objective(&model, &cache, &order, Some(&mut gradient))?.0;
        Some(head::linear_gap(&model, &gradient)?)
    };
    let max_column_l1 = model.validate()?;
    let diagnostic_seconds = audit_at.elapsed().as_secs_f64();
    checkpoint::save(
        &model,
        BufWriter::new(File::create(directory.join("model.bin"))?),
    )?;
    let model_hash = checkpoint::digest(BufReader::new(File::open(directory.join("model.bin"))?))?;
    write_json(
        directory.join("result.json"),
        &json!({"status":"training_check_complete","heldout_scored":false,
        "source_fingerprint":fingerprint(),"producer_commit":commit,"task":task_name,"depth":depth,"mode":mode,
        "initializer_channel":initializer_channel,"initializer_train_ce":initial_ce,"train_ce":ce,"radius":model.radius,
        "max_class_l1":max_column_l1,"full_gradient_linear_gap":gap,"model_sha256":model_hash,
        "gradient_updates":updates,"gradient_examples_seen":examples_seen,"gradient_prefixes_seen":prefixes_seen,
        "initializer_examples_seen":train.len(),"initializer_prefixes_seen":n,"parameters":model.coefficients.len(),
        "timing_seconds":{"data":data_seconds,"features":feature_seconds,"initializer":initializer_seconds,
            "optimization":optimization_seconds,"final_diagnostic":diagnostic_seconds,"total":start.elapsed().as_secs_f64()}}),
    )?;
    Ok(())
}

pub fn replay(directory: &Path) -> Result<Value> {
    let metadata: Value = serde_json::from_reader(File::open(directory.join("metadata.json"))?)?;
    let result: Value = serde_json::from_reader(File::open(directory.join("result.json"))?)?;
    if metadata["source_fingerprint"] != fingerprint()
        || result["source_fingerprint"] != fingerprint()
        || result["status"] != "training_check_complete"
        || result["heldout_scored"] != false
        || metadata["protocol"] != serde_json::from_str::<Value>(PROTOCOL)?
        || ["task", "depth", "mode", "producer_commit"]
            .iter()
            .any(|key| result[key] != metadata[key])
    {
        return Err("training replay provenance mismatch".into());
    }
    let hash = checkpoint::digest(BufReader::new(File::open(directory.join("model.bin"))?))?;
    if result["model_sha256"] != hash {
        return Err("model hash mismatch".into());
    }
    let model = checkpoint::load(BufReader::new(File::open(directory.join("model.bin"))?))?;
    let task_name = metadata["task"].as_str().ok_or("missing replay task")?;
    let row = meta::selected(task_name)?;
    let task = data::build(&row)?;
    if metadata["data_fingerprints"] != row["data_fingerprints"]
        || model.bank.vocab != task.vocab
        || metadata["bank"] != serde_json::to_value(&model.bank)?
        || model.radius
            != metadata["protocol"]["radius_times_vocab"].as_f64().unwrap() * task.vocab as f64
    {
        return Err("saved model or data configuration mismatch".into());
    }
    let expected_bank = Bank::discover(
        task.vocab,
        metadata["depth"].as_u64().ok_or("missing depth")? as usize,
        metadata["protocol"]["history_slots"].as_u64().unwrap() as usize,
        Transport::Fourier,
        &task.splits["train"],
    )?;
    if metadata["bank"] != serde_json::to_value(expected_bank)? {
        return Err("training-only feature discovery does not replay".into());
    }
    let ce = convex_stress_hull::teacher_ce(&model, &task.splits["train"])?;
    let recorded = result["train_ce"].as_f64().ok_or("missing recorded CE")?;
    if (ce - recorded).abs() > 1e-10 {
        return Err("saved training CE does not replay".into());
    }
    Ok(
        json!({"status":"training_replay_pass","heldout_scored":false,"train_ce":ce,
        "ce_difference":ce-recorded,"model_sha256":hash,"max_class_l1":model.validate()?}),
    )
}

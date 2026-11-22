//! Fixed-budget comparisons. Crossing the target records an observation and training continues.
use super::*;
use crate::head::Head;
use std::io::{BufRead, BufReader};

fn exact(engine: &mut Engine, examples: &[Example]) -> Result<u64> {
    metrics::evaluate(engine, examples, true)?["exact"]
        .as_u64()
        .ok_or_else(|| "missing exact-sequence score".into())
}

fn score_checkpoint(cfg: Config, task: &Task, bytes: &[u8]) -> Result<(Value, Option<Value>)> {
    if bytes.len() != layout::total_count(&cfg) * 4 {
        return Err("fixed-budget checkpoint shape mismatch".into());
    }
    let weights: Vec<_> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    layout::validate(&cfg, &weights)?;
    let mut engine = Engine::new(cfg, &weights)?;
    let mut scores = BTreeMap::new();
    for (split, examples) in &task.splits {
        if split == &task.selection || split == "test" || split.starts_with("test/") {
            scores.insert(
                split,
                json!({"correct":exact(&mut engine, examples)?,"total":examples.len()}),
            );
        }
    }
    let causality = if matches!(
        engine.config.head,
        Head::Transition
            | Head::Recurrent
            | Head::Bounded
            | Head::InputProduct
            | Head::PhasePrefix
            | Head::InputPhase
            | Head::AffinePhase
            | Head::BilinearPhase
            | Head::SharedBilinear
            | Head::ReflectionPhase
            | Head::SingleReflection
            | Head::ContextRotation
            | Head::CrossedContext
            | Head::AssociativeReflection
            | Head::PeriodicReflection
    ) {
        let sample = task.splits[&task.selection]
            .iter()
            .max_by_key(|e| e.tokens.len())
            .ok_or("missing held-out sequence for causality check")?;
        let tokens: Vec<_> = sample.tokens.iter().map(|&v| v as u32).collect();
        Some(memory_checks::trained_prefixes(&mut engine, &tokens)?)
    } else {
        None
    };
    Ok((json!(scores), causality))
}

pub enum Source<'a> {
    ComparedHead(&'a str),
    Catalog,
}

/// Hash the ordered input sequences alone. For generative tasks use each
/// prompt, and for categorical tasks use its input tokens. Lengths delimit rows.
fn input_digest(examples: &[Example]) -> String {
    use sha2::{Digest, Sha256};
    use std::fmt::Write as _;
    let mut digest = Sha256::new();
    digest.update((examples.len() as u64).to_le_bytes());
    for e in examples {
        let input = if e.prompt.is_empty() {
            &e.tokens
        } else {
            &e.prompt
        };
        digest.update((input.len() as u64).to_le_bytes());
        for token in input {
            digest.update(token.to_le_bytes());
        }
    }
    let mut hex = String::with_capacity(64);
    for byte in digest.finalize() {
        write!(hex, "{byte:02x}").unwrap();
    }
    hex
}

pub fn run(source: Source, name: &str, seed: u32, updates: usize, out: &Path) -> Result<()> {
    let result = measure(source, name, seed, updates, out, true)?;
    println!("{result}");
    Ok(())
}

fn directory(out: &Path, arm: &str, name: &str, seed: u32, updates: usize) -> std::path::PathBuf {
    out.join(arm)
        .join(name)
        .join(format!("seed{seed}-{updates}"))
}

/// Replay the frozen catalog record and return the very bytes it authenticates.
pub(super) fn replay_catalog(
    name: &str,
    seed: u32,
    updates: usize,
    out: &Path,
) -> Result<(Value, Vec<u8>)> {
    let result = measure(Source::Catalog, name, seed, updates, out, false)?;
    let bytes = fs::read(directory(out, MODEL, name, seed, updates).join("weights.f32"))?;
    if result["weights_sha256"] != hash(&bytes) {
        return Err("catalog weights changed after replay".into());
    }
    Ok((result, bytes))
}

fn measure(
    source: Source,
    name: &str,
    seed: u32,
    updates: usize,
    out: &Path,
    train_missing: bool,
) -> Result<Value> {
    let (head, model, arm, (row, mut task)) = match source {
        Source::ComparedHead(arm) => {
            let head = Head::parse(arm)?;
            (head, head.id(), arm, load(name)?)
        }
        Source::Catalog => (architecture::HEAD, MODEL, MODEL, catalog::load(name)?),
    };
    let schedule = row["case"]["max_updates"].as_u64().unwrap() as usize;
    let observe = row["case"]["observe"].as_u64().unwrap() as usize;
    let batch = row["case"]["batch"].as_u64().unwrap() as usize;
    if updates == 0
        || updates > schedule
        || (updates != schedule && !updates.is_multiple_of(observe))
    {
        return Err("budget must end at a scheduled observation within the task horizon".into());
    }
    let mut cfg = task_config(&row, &task, true)?;
    cfg.head = head;
    let mut protocol = json!({"model":model, "case":name, "seed":seed, "updates":updates,
        "schedule_updates":schedule, "observe":observe, "batch":batch, "selection":task.selection,
        "train_examples":task.splits["train"].len(), "heldout_examples":task.splits[&task.selection].len(),
        "suite_sha256":suite_hash(&row), "kernite_version":KERNITE_VERSION,
        "fp32_affines":precision::FP32_AFFINE_REFERENCE, "bf16":precision::BF16,
        "tf32_backward":precision::TF32_BACKWARD, "parameters":layout::independent_count(&cfg),
        "stored_parameters":layout::total_count(&cfg), "head_parameters":head.parameters(&cfg)});
    let mut initialization = None;
    if matches!(source, Source::Catalog) {
        let weights = layout::initialize_model(&cfg, seed);
        let bytes: Vec<_> = weights.iter().flat_map(|w| w.to_le_bytes()).collect();
        protocol["initial_weights_sha256"] = json!(hash(&bytes));
        protocol["shape"] =
            json!({"width":cfg.width,"layers":cfg.layers,"heads":cfg.heads,"vocabulary":cfg.vocab});
        protocol["ordered_inputs_sha256"] = json!({
            "train":input_digest(&task.splits["train"]),
            "held_out":input_digest(&task.splits[&task.selection])
        });
        initialization = Some(weights);
    }
    let dir = directory(out, arm, name, seed, updates);
    let result_file = dir.join("result.json");
    if train_missing {
        fs::create_dir_all(&dir)?;
    } else if !result_file.is_file() {
        return Err("catalog comparison requires a completed fixed-budget cell".into());
    }
    let lock = fs::File::options()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(dir.join("cell.lock"))?;
    lock.lock()?;
    let weights_file = dir.join("weights.f32");
    let history_file = dir.join("observations.jsonl");
    let saved: Option<Value> = if result_file.try_exists()? {
        let result: Value = serde_json::from_slice(&fs::read(&result_file)?)?;
        if result["protocol"] != protocol {
            return Err("saved benchmark has a different protocol".into());
        }
        Some(result)
    } else if train_missing {
        None
    } else {
        return Err("catalog result disappeared before replay".into());
    };
    let gpu = if let Some(result) = &saved {
        result["gpu"].clone()
    } else {
        let weights = initialization.unwrap_or_else(|| layout::initialize_model(&cfg, seed));
        let mut e = Engine::new(cfg, &weights)?;
        let mut training = Training::new(&e, seed, schedule)?;
        let held = &task.splits[&task.selection];
        let mut history = fs::File::create(&history_file)?;
        for step in 0..=updates {
            if step.is_multiple_of(observe) || step == updates {
                let point = json!({"step":step,"correct":exact(&mut e, held)?,"examples_seen":training.seen});
                writeln!(history, "{point}")?;
            }
            if step < updates {
                training.update(&mut e, &task.splits["train"], batch, step)?;
            }
        }
        history.sync_all()?;
        let values = e.parameters.read()?;
        layout::validate(&e.config, &values)?;
        let bytes: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        publish(&dir, "weights.f32", &bytes)?;
        json!(e.cuda.name())
    };
    if matches!(name, "basis-parity16" | "basis-parity20" | "basis-parity24") {
        catalog::add_parity_probes(&mut task)?;
    }
    let mut cfg = task_config(&row, &task, false)?;
    cfg.head = head;
    let bytes = fs::read(weights_file)?;
    let (scores, causality) = score_checkpoint(cfg, &task, &bytes)?;
    let train = task.splits["train"].len();
    let held = task.splits[&task.selection].len() as u64;
    let epoch = train.div_ceil(batch);
    let steps = (0..updates)
        .step_by(observe)
        .chain(std::iter::once(updates));
    let mut history = BufReader::new(fs::File::open(history_file)?).lines();
    let (mut first_99, mut initial, mut last) = (None, 0, 0);
    for step in steps {
        let point: Value = serde_json::from_str(&history.next().ok_or("missing observation")??)?;
        last = point["correct"]
            .as_u64()
            .filter(|&n| n <= held)
            .ok_or("invalid score")?;
        if point["step"] != step
            || point["examples_seen"] != step / epoch * train + step % epoch * batch
        {
            return Err("benchmark observation cadence or batch order count mismatch".into());
        }
        if step == 0 {
            initial = last;
        }
        if last * 100 >= 99 * held {
            first_99.get_or_insert(step);
        }
    }
    if history.next().is_some() || scores[&task.selection]["correct"] != last {
        return Err("history failed final held-out checkpoint replay".into());
    }
    let mut result = json!({"protocol":protocol, "initial_correct":initial, "first_99":first_99,
        "scores":scores, "weights_sha256":hash(&bytes), "gpu":gpu});
    if let Some(probe) = causality {
        result["causality"] = probe;
    }
    if let Some(saved) = saved {
        if result != saved {
            return Err("fixed-budget result differs from independent replay".into());
        }
    } else {
        publish(&dir, "result.json", &serde_json::to_vec_pretty(&result)?)?;
    }
    Ok(result)
}

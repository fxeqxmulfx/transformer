//! Train and score through the engine's public methods, including generation.
//! Model configuration sees vocabulary and context, never a case name or a target.
use super::*;
use data::{Example, Task};
use std::collections::BTreeMap;

pub mod alignment;
pub mod benchmark;
mod catalog;
mod language_probes;
pub mod report;

fn cases() -> Result<Vec<Value>> {
    let suite: Value = serde_json::from_str(arithmetic::ORIGINAL35)?;
    Ok(suite["tasks"].as_array().ok_or("missing tasks")?.clone())
}

fn task_config(row: &Value, task: &Task, reserve_training_batch: bool) -> Result<Config> {
    let c = &row["case"];
    let number = |key: &str| c[key].as_u64().map(|x| x as usize).ok_or("missing shape");
    let mut cfg = config(number("batch")?)?;
    cfg.width = number("width")?;
    cfg.heads = number("heads")?;
    cfg.layers = number("layers")?;
    let evaluation = if cfg.width >= 128 && cfg.layers >= 6 {
        32
    } else {
        128
    };
    // The scorer visits every example independently. Bound its allocation on
    // smaller GPUs while preserving the suite minibatch and its update order.
    let cap = env::var("PHASOR_EVAL_BATCH_CAP")
        .ok()
        .map(|s| s.parse::<std::num::NonZeroUsize>())
        .transpose()?
        .map_or(evaluation, |n| n.get().min(evaluation));
    cfg.max_batch = if reserve_training_batch {
        cfg.max_batch.max(cap)
    } else {
        cap
    };
    cfg.gain = cfg.heads as f32 / cfg.width as f32;
    cfg.warmup = schedule::Schedule::new(number("max_updates")?)?.warmup;
    cfg.vocab = task.vocab;
    cfg.max_length = task
        .splits
        .values()
        .flatten()
        .map(|e| {
            e.tokens
                .len()
                .max(e.prompt.len() + e.limit.saturating_sub(1))
        })
        .max()
        .ok_or("empty task")?;
    cfg.head = architecture::HEAD;
    Ok(cfg)
}

fn nv11_case() -> Value {
    json!({"name":"arithmetic-cubic-product","tier":"NV11","train_examples":4704,
        "heldout_examples":4705,"selection":"validation",
        "case":{"width":128,"heads":4,"layers":1,"batch":BATCH,"max_updates":BUDGET,"observe":OBSERVE}})
}

fn nv11() -> Result<(Value, Task)> {
    Ok((nv11_case(), arithmetic_task(cubic_data()?)))
}

fn arithmetic_task(d: dataset::Dataset) -> Task {
    let examples = |rows: Vec<[u32; 3]>| {
        rows.into_iter()
            .map(|[a, b, y]| Example {
                task: "arithmetic".into(),
                tokens: vec![i64::from(a), i64::from(b)],
                targets: vec![-100, i64::from(y)],
                changes: vec![false; 2],
                prompt: vec![],
                answer: vec![],
                limit: 0,
                field: None,
                row_index: None,
            })
            .collect()
    };
    Task {
        vocab: d.training.vocabulary,
        selection: "validation".into(),
        splits: BTreeMap::from([
            ("train".into(), examples(d.training.rows)),
            ("validation".into(), examples(d.validation)),
        ]),
    }
}

use architecture::MODEL;

pub const EXTENSIONS: &str = include_str!("../extensions.tsv");

pub fn extensions(registry: &str) -> Result<()> {
    for (name, tier) in catalog::registered(registry)? {
        println!("{name}\t{tier}");
    }
    Ok(())
}

pub fn routes() -> Result<()> {
    for row in cases()? {
        let task = data::build(&row)?;
        let cfg = task_config(&row, &task, true)?;
        println!(
            "{}\t{}\t{}\t{}\t{}",
            row["name"].as_str().unwrap(),
            row["tier"].as_str().unwrap(),
            cfg.max_length,
            cfg.vocab,
            cfg.head == architecture::HEAD
        );
    }
    Ok(())
}

fn load(name: &str) -> Result<(Value, Task)> {
    if name == "arithmetic-cubic-product" {
        return nv11();
    }
    if let Some(row) = cases()?.into_iter().find(|r| r["name"] == name) {
        let task = data::build(&row)?;
        return Ok((row, task));
    }
    catalog::load(name)
}

fn suite_hash(row: &Value) -> String {
    row["catalog_sha256"]
        .as_str()
        .map_or_else(|| hash(arithmetic::ORIGINAL35.as_bytes()), str::to_owned)
}

// All checkpoints predating this field used the original pinned Kernite 0.3.2.
fn trained_with(result: &Value) -> Result<&str> {
    result
        .get("kernite_version")
        .map_or(Ok("0.3.2"), |version| {
            version
                .as_str()
                .ok_or_else(|| "invalid recorded Kernite version".into())
        })
}

fn publish(dir: &Path, name: &str, bytes: &[u8]) -> Result<()> {
    let pending = dir.join(format!("{name}.pending"));
    let mut file = fs::File::create(&pending)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    fs::rename(pending, dir.join(name))?;
    fs::File::open(dir)?.sync_all()?;
    Ok(())
}

pub fn run(name: &str, seed: u32, out: &Path, cutoff: Option<usize>) -> Result<()> {
    let started = std::time::Instant::now();
    let (row, task) = load(name)?;
    let cfg = task_config(&row, &task, true)?;
    let batch = row["case"]["batch"].as_u64().unwrap() as usize;
    let budget = row["case"]["max_updates"].as_u64().unwrap() as usize;
    let observe = row["case"]["observe"].as_u64().unwrap() as usize;
    if cutoff.is_some_and(|n| n == 0 || n > budget || (n != budget && !n.is_multiple_of(observe))) {
        return Err(
            "screen cutoff must be a scheduled observation within the original budget".into(),
        );
    }
    let train = &task.splits["train"];
    let held = &task.splits[&task.selection];
    let dest = out.join(name).join(format!("seed{seed}"));
    fs::create_dir_all(&dest)?;
    let lock = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(dest.join("cell.lock"))?;
    let waiting = std::time::Instant::now();
    lock.lock()
        .map_err(|err| format!("cannot lock {}: {err}", dest.display()))?;
    let waited = waiting.elapsed();
    if dest.join("result.json").exists() {
        if let Some(limit) = cutoff {
            let saved: Value = serde_json::from_slice(&fs::read(dest.join("result.json"))?)?;
            report::validate(&row, seed, &dest, &saved)?;
            if saved["observation"]["step"].as_u64().unwrap() > limit as u64 {
                use std::io::{BufRead, BufReader};
                let lines =
                    BufReader::new(fs::File::open(dest.join("observations.jsonl"))?).lines();
                for line in lines {
                    let point: Value = serde_json::from_str(&line?)?;
                    if point["step"] == limit {
                        println!(
                            "{}",
                            json!({"model":MODEL,"status":"screen cutoff","observation":point,
                                "kernite_version":trained_with(&saved)?})
                        );
                        return Ok(());
                    }
                }
                return Err("scheduled cutoff missing from saved observation log".into());
            }
        }
        return replay(name, seed, out);
    }
    let parameters = layout::initialize_model(&cfg, seed);
    let mut e = Engine::new(cfg, &parameters)?;
    let mut training = Training::new(&e, seed, budget)?;
    let mut previous = 0;
    // Restart an interrupted cell from its fixed seed and publish result.json
    // last. A completed cell is verified above before any file can change.
    let mut history = fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(dest.join("observations.jsonl"))?;
    for step in 0..=budget {
        if step % observe == 0 || step == budget {
            let scores = metrics::evaluate(&mut e, held, true)?;
            let correct = scores["exact"].as_u64().ok_or("missing exact count")? as usize;
            let pass = correct * 100 >= 99 * held.len();
            let point = json!({"case":name,"tier":row["tier"],"seed":seed,"step":step,
                "examples_seen":training.seen,"correct":correct,"total":held.len(),"pass":pass});
            writeln!(history, "{point}")?;
            history.flush()?;
            if pass || step == budget {
                let training = metrics::evaluate(&mut e, train, false)?;
                let values = e.parameters.read()?;
                layout::validate(&e.config, &values)?;
                let bytes: Vec<_> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
                history.sync_all()?;
                publish(&dest, "weights.f32", &bytes)?;
                let mut result = json!({"model":MODEL,"observation":point,"previous_step":previous,"train_correct":training["exact"],
                    "train_total":train.len(),"selection":task.selection,
                    "parameters":values.len(),"weights_sha256":hash(&bytes),"gpu":e.cuda.name(),
                    "fp32_affines":precision::FP32_AFFINE_REFERENCE,"suite_sha256":suite_hash(&row),"kernite_version":KERNITE_VERSION,
                    "elapsed_seconds":(started.elapsed()-waited).as_secs_f64()});
                architecture::record(&mut result, &e.config, &values);
                publish(&dest, "result.json", &serde_json::to_vec_pretty(&result)?)?;
                println!("{result}");
                return Ok(());
            }
            if cutoff == Some(step) {
                println!(
                    "{}",
                    json!({"model":MODEL,"status":"screen cutoff","observation":point,"kernite_version":KERNITE_VERSION})
                );
                return Ok(());
            }
            previous = step;
        }
        training.update(&mut e, train, batch, step)?;
    }
    unreachable!()
}

/// Both stopping protocols consume the same shuffled epochs, including short final batches.
struct Training {
    optimizer: optimizer::Optimizer,
    rng: random::Random,
    epoch: Vec<u32>,
    offset: usize,
    seen: usize,
    packed: metrics::Packed,
}

impl Training {
    fn new(engine: &Engine, seed: u32, schedule_budget: usize) -> Result<Self> {
        Ok(Self {
            optimizer: optimizer::Optimizer::new(engine, schedule_budget)?,
            rng: random::Random::new(seed.checked_add(10_000).ok_or("seed overflow")?),
            epoch: Vec::new(),
            offset: 0,
            seen: 0,
            packed: metrics::Packed::default(),
        })
    }

    fn update(
        &mut self,
        engine: &mut Engine,
        train: &[Example],
        batch: usize,
        step: usize,
    ) -> Result<()> {
        if self.offset == self.epoch.len() {
            self.epoch = (0..train.len() as u32).collect();
            self.rng.shuffle(&mut self.epoch);
            self.offset = 0;
        }
        let end = (self.offset + batch).min(self.epoch.len());
        let indices = &self.epoch[self.offset..end];
        self.packed.fill(train, indices);
        engine.forward(
            &self.packed.tokens,
            &self.packed.labels,
            indices.len(),
            self.packed.length,
        )?;
        engine.backward(indices.len(), self.packed.length)?;
        self.optimizer.step(engine, step)?;
        self.seen += indices.len();
        self.offset = end;
        Ok(())
    }
}

/// Re-score a saved model using the pinned split and decoder. For pair tasks
/// also evaluate the zero row with the fitted tables, then with all tables reset.
pub fn replay(name: &str, seed: u32, out: &Path) -> Result<()> {
    score_checkpoint(name, seed, out, ProbeSet::Validation)
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ProbeSet {
    Validation,
    Tests,
    ExtendedLanguage,
}

pub fn score_checkpoint(name: &str, seed: u32, out: &Path, probes: ProbeSet) -> Result<()> {
    let test_splits = probes != ProbeSet::Validation;
    let extended = probes == ProbeSet::ExtendedLanguage;
    if extended && !name.starts_with("language-l") {
        return Err("extended probes require a source block language".into());
    }
    let (row, mut task) = load(name)?;
    if test_splits && matches!(name, "basis-parity16" | "basis-parity20" | "basis-parity24") {
        catalog::add_parity_probes(&mut task)?;
    }
    if test_splits && name.starts_with("language-l") {
        language_probes::add(&mut task, name, extended).map_err(|err| err.to_string())?;
    }
    let cfg = task_config(&row, &task, !extended)?;
    let dest = out.join(name).join(format!("seed{seed}"));
    let result: Value = serde_json::from_slice(&fs::read(dest.join("result.json"))?)?;
    report::validate(&row, seed, &dest, &result)?;
    let bytes = fs::read(dest.join("weights.f32"))?;
    if result["model"] != MODEL
        || result["weights_sha256"] != hash(&bytes)
        || result["suite_sha256"] != suite_hash(&row)
        || result["fp32_affines"] != precision::FP32_AFFINE_REFERENCE
        || result["observation"]["case"] != name
        || result["observation"]["seed"] != seed
        || bytes.len() != 4 * layout::total_count(&cfg)
    {
        return Err("checkpoint identity, shape, or precision mismatch".into());
    }
    let mut values: Vec<_> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect();
    let mut e = Engine::new(cfg, &values)?;
    let held = &task.splits[&task.selection];
    let correct = metrics::evaluate(&mut e, held, true)?["exact"].clone();
    if correct != result["observation"]["correct"] || result["observation"]["total"] != held.len() {
        return Err("checkpoint failed to reproduce its held-out score".into());
    }
    let mut report = json!({"case":name,"seed":seed,"correct":correct,"total":held.len(),
        "observation":result["observation"],
        "model":MODEL,"weights_sha256":result["weights_sha256"],
        "kernite_version":trained_with(&result)?,"scored_with_kernite":KERNITE_VERSION});
    if test_splits {
        let mut scores = BTreeMap::new();
        for (split, examples) in &task.splits {
            if split == "test" || split.starts_with("test/") {
                let measured = metrics::evaluate(&mut e, examples, true)?;
                if split == language_probes::STUTTER {
                    report["first_stutter_error"] = if measured["exact"] == examples.len() {
                        Value::Null
                    } else {
                        language_probes::stutter_error(&mut e, examples)?
                    };
                }
                scores.insert(
                    split,
                    json!({"correct":measured["exact"],"total":examples.len()}),
                );
            }
        }
        report["test_splits"] = json!(scores);
        if let Some(words) = task.splits.get(language_probes::EXHAUSTIVE) {
            report["shortest_binary_error"] = language_probes::shortest_error(&mut e, words)?;
        }
    }
    let zeros: Vec<_> = held
        .iter()
        .filter(|e| e.tokens.len() == 2 && e.prompt.is_empty() && e.tokens[0] == 0)
        .cloned()
        .collect();
    if !zeros.is_empty() {
        report["zero_row_total"] = json!(zeros.len());
        report["zero_row_correct"] = metrics::evaluate(&mut e, &zeros, true)?["exact"].clone();
        if !architecture::RAW {
            let start = pair_memory::start(&e.config);
            values[start..pair_memory::gain(&e.config)].fill(0.);
            e.parameters.write(&values)?;
            report["zero_row_correct_without_tables"] =
                metrics::evaluate(&mut e, &zeros, true)?["exact"].clone();
        }
    }
    println!("{report}");
    Ok(())
}

//! Train and score through the engine's public methods, including generation.
//! Model configuration sees vocabulary and context, never a case name or a target.
use super::*;
use data::{Example, Task};
use std::collections::BTreeMap;

fn cases() -> Result<Vec<Value>> {
    let suite: Value = serde_json::from_str(arithmetic::ORIGINAL35)?;
    Ok(suite["tasks"].as_array().ok_or("missing tasks")?.clone())
}

fn task_config(row: &Value, task: &Task) -> Result<Config> {
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
    cfg.max_batch = cfg.max_batch.max(evaluation);
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
    cfg.shared_values = true;
    Ok(cfg)
}

fn nv11() -> Result<(Value, Task)> {
    let d = cubic_data()?;
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
    let row = json!({"name":"arithmetic-cubic-product","tier":"NV11",
        "case":{"width":128,"heads":4,"layers":1,"batch":BATCH,"max_updates":BUDGET,"observe":OBSERVE}});
    Ok((
        row,
        Task {
            vocab: d.training.vocabulary,
            selection: "validation".into(),
            splits: BTreeMap::from([
                ("train".into(), examples(d.training.rows)),
                ("validation".into(), examples(d.validation)),
            ]),
        },
    ))
}

const MODEL: &str = "phasor-causal-value-residual-v1";

pub fn routes() -> Result<()> {
    for row in cases()? {
        let task = data::build(&row)?;
        let cfg = task_config(&row, &task)?;
        println!(
            "{}\t{}\t{}\t{}\t{}",
            row["name"].as_str().unwrap(),
            row["tier"].as_str().unwrap(),
            cfg.max_length,
            cfg.vocab,
            cfg.shared_values
        );
    }
    Ok(())
}

fn load(name: &str) -> Result<(Value, Task)> {
    if name == "arithmetic-cubic-product" {
        return nv11();
    }
    let row = cases()?
        .into_iter()
        .find(|r| r["name"] == name)
        .ok_or("unknown suite case")?;
    let task = data::build(&row)?;
    Ok((row, task))
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

pub fn run(name: &str, seed: u32, out: &Path) -> Result<()> {
    let (row, task) = load(name)?;
    let cfg = task_config(&row, &task)?;
    let extra = pair_memory::parameters(&cfg);
    let batch = row["case"]["batch"].as_u64().unwrap() as usize;
    let budget = row["case"]["max_updates"].as_u64().unwrap() as usize;
    let observe = row["case"]["observe"].as_u64().unwrap() as usize;
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
    lock.try_lock()
        .map_err(|err| format!("cannot lock {}: {err}", dest.display()))?;
    if dest.join("result.json").exists() {
        return replay(name, seed, out);
    }
    let parameters = layout::initialize_model(&cfg, seed);
    let mut e = Engine::new(cfg, &parameters)?;
    let mut opt = optimizer::Optimizer::new(&e, budget)?;
    let mut rng = random::Random::new(seed.checked_add(10000).ok_or("seed overflow")?);
    let mut epoch = Vec::new();
    let (mut offset, mut seen, mut previous) = (0, 0, 0);
    let mut packed = metrics::Packed::default();
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
                "examples_seen":seen,"correct":correct,"total":held.len(),"pass":pass});
            writeln!(history, "{point}")?;
            history.flush()?;
            if pass || step == budget {
                let training = metrics::evaluate(&mut e, train, false)?;
                let values = e.parameters.read()?;
                layout::validate(&e.config, &values)?;
                let bytes: Vec<_> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
                history.sync_all()?;
                publish(&dest, "weights.f32", &bytes)?;
                let result = json!({"model":MODEL,"gain":values[pair_memory::gain(&e.config)],"observation":point,"previous_step":previous,"train_correct":training["exact"],
                    "train_total":train.len(),"selection":task.selection,"pair_parameters":extra,
                    "parameters":values.len(),"weights_sha256":hash(&bytes),"gpu":e.cuda.name(),
                    "fp32_affines":precision::FP32_AFFINE_REFERENCE,"suite_sha256":hash(arithmetic::ORIGINAL35.as_bytes())});
                publish(&dest, "result.json", &serde_json::to_vec_pretty(&result)?)?;
                println!("{result}");
                return Ok(());
            }
            previous = step;
        }
        if offset == epoch.len() {
            epoch = (0..train.len() as u32).collect();
            rng.shuffle(&mut epoch);
            offset = 0;
        }
        let end = (offset + batch).min(epoch.len());
        let indices = &epoch[offset..end];
        packed.fill(train, indices);
        e.forward(&packed.tokens, &packed.labels, indices.len(), packed.length)?;
        e.backward(indices.len(), packed.length)?;
        opt.step(&e, step)?;
        seen += indices.len();
        offset = end;
    }
    unreachable!()
}

/// Re-score a saved model using the pinned split and decoder. For pair tasks
/// also evaluate the zero row with the fitted tables, then with all tables reset.
pub fn replay(name: &str, seed: u32, out: &Path) -> Result<()> {
    let (row, task) = load(name)?;
    let cfg = task_config(&row, &task)?;
    let dest = out.join(name).join(format!("seed{seed}"));
    let result: Value = serde_json::from_slice(&fs::read(dest.join("result.json"))?)?;
    let bytes = fs::read(dest.join("weights.f32"))?;
    if result["model"] != MODEL
        || result["weights_sha256"] != hash(&bytes)
        || result["suite_sha256"] != hash(arithmetic::ORIGINAL35.as_bytes())
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
    let mut report = json!({"case":name,"seed":seed,"correct":correct,"total":held.len()});
    let zeros: Vec<_> = held
        .iter()
        .filter(|e| e.tokens.len() == 2 && e.prompt.is_empty() && e.tokens[0] == 0)
        .cloned()
        .collect();
    if !zeros.is_empty() {
        report["zero_row_total"] = json!(zeros.len());
        report["zero_row_correct"] = metrics::evaluate(&mut e, &zeros, true)?["exact"].clone();
        let start = pair_memory::start(&e.config);
        values[start..pair_memory::gain(&e.config)].fill(0.);
        e.parameters.write(&values)?;
        report["zero_row_correct_without_tables"] =
            metrics::evaluate(&mut e, &zeros, true)?["exact"].clone();
    }
    println!("{report}");
    Ok(())
}

//! Frozen GPT stress cases and schedule for Argand and its residual controls.
use super::{Example, Result, Task, Value, fs, hash, json, load, publish, suite_hash, task_config};
use crate::{
    Config, KERNITE_VERSION,
    phasor_v2::{AmsGradWConfig, Engine, Variant},
    phasor_v2_metrics as metrics,
    random::Random,
    schedule::Schedule,
};
use std::path::{Path, PathBuf};

pub fn command(op: &str, args: &[String]) -> Result<()> {
    let (family, action) = op.split_once('-').ok_or("missing decoder operation")?;
    let variant = if family == "argand" {
        Variant::argand()?
    } else {
        Variant::v2()?
    };
    match (action, args) {
        ("check", []) => crate::phasor_v2::check(variant),
        ("run", [name, seed, repeat, out, cutoff @ ..]) if cutoff.len() <= 1 => run(
            name,
            seed.parse()?,
            repeat.parse()?,
            Path::new(out),
            cutoff.first().map(|step| step.parse()).transpose()?,
            variant,
        ),
        ("replay", [name, seed, repeat, out]) => {
            replay(name, seed.parse()?, repeat.parse()?, Path::new(out), variant)
        }
        _ => Err(format!(
            "usage: {family}-check | {family}-run CASE SEED REPEAT RUN_DIR [LAST_OBSERVATION] | {family}-replay CASE SEED REPEAT RUN_DIR"
        )
        .into()),
    }
}

struct Cell {
    row: Value,
    task: Task,
    shape: Config,
    batch: usize,
    every: usize,
    schedule: Schedule,
    directory: PathBuf,
    repeat: usize,
    seed: u32,
    variant: Variant,
}

impl Cell {
    fn new(name: &str, seed: u32, repeat: usize, out: &Path, variant: Variant) -> Result<Self> {
        if seed > 2 || repeat == 0 {
            return Err("expected seed 0 through 2 and a positive repeat count".into());
        }
        let (row, task) = load(name)?;
        if !matches!(row["tier"].as_str(), Some("short" | "long")) {
            return Err("expected a case from the canonical GPT tiers".into());
        }
        let number = |k: &str| {
            row["case"][k]
                .as_u64()
                .ok_or("missing case setting")
                .map(|n| n as usize)
        };
        let shape = task_config(&row, &task, true)?;
        let (batch, every, schedule) = (
            number("batch")?,
            number("observe")?,
            Schedule::new(number("max_updates")?)?,
        );
        let directory = out.join(name).join(format!("seed{seed}"));
        Ok(Self {
            row,
            task,
            shape,
            batch,
            every,
            schedule,
            directory,
            repeat,
            seed,
            variant,
        })
    }

    fn engine(&self) -> Result<Engine> {
        let mut engine = Engine::new(self.shape.clone(), self.repeat, self.seed, self.variant)?;
        engine.config.max_batch = task_config(&self.row, &self.task, false)?.max_batch;
        Ok(engine)
    }

    fn identity(&self) -> Value {
        let mut identity = json!({"model":self.variant.model(),"repeat":self.repeat,"seed":self.seed,"case":self.row["name"],
            "tier":self.row["tier"],"suite_sha256":suite_hash(&self.row),"kernite":KERNITE_VERSION});
        if let Some(config) = self.variant.amsgradw() {
            if config.sqrt_scale != 1.0 {
                identity["amsgradw_sqrt_scale"] = config.sqrt_scale.to_string().into();
            }
            if config.beta2 != AmsGradWConfig::DEFAULT_BETA2 {
                identity["amsgradw_beta2"] = config.beta2.to_string().into();
            }
        }
        identity
    }

    fn state_file(&self, step: usize) -> PathBuf {
        self.directory.join(format!("state-{step}.f32"))
    }

    fn checkpoint(&self) -> Result<Option<(Value, Vec<f32>)>> {
        let path = self.directory.join("result.json");
        if !path.try_exists()? {
            return Ok(None);
        }
        let mut record: Value = serde_json::from_slice(&fs::read(path)?)?;
        let step = record["step"].as_u64().ok_or("missing update number")? as usize;
        if let Some(legacy) = self.variant.legacy_model()
            && record["identity"]["model"] == legacy
        {
            record["identity"]["model"] = self.variant.model().into();
        }
        if record["identity"] != self.identity() || step > self.schedule.budget {
            return Err("checkpoint belongs to another decoder or suite cell".into());
        }
        let bytes = fs::read(self.state_file(step))?;
        if record["state_sha256"] != hash(&bytes) || !bytes.len().is_multiple_of(4) {
            return Err("invalid decoder checkpoint digest".into());
        }
        let state = bytes
            .chunks_exact(4)
            .map(|w| f32::from_le_bytes(w.try_into().unwrap()))
            .collect();
        Ok(Some((record, state)))
    }

    fn save(&self, engine: &Engine, step: usize, record: Value) -> Result<()> {
        let bytes: Vec<_> = engine
            .state()?
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        let filename = format!("state-{step}.f32");
        publish(&self.directory, &filename, &bytes)?;
        let mut report = record;
        report["state_sha256"] = hash(&bytes).into();
        publish(
            &self.directory,
            "result.json",
            &serde_json::to_vec_pretty(&report)?,
        )?;
        println!("{report}");
        Ok(())
    }
}

fn run(
    name: &str,
    seed: u32,
    repeat: usize,
    out: &Path,
    cutoff: Option<usize>,
    variant: Variant,
) -> Result<()> {
    let cell = Cell::new(name, seed, repeat, out, variant)?;
    let budget = cell.schedule.budget;
    let until = cutoff.unwrap_or(budget);
    if until == 0 || until > budget || (until != budget && !until.is_multiple_of(cell.every)) {
        return Err("cutoff must be a scheduled observation within the frozen budget".into());
    }
    fs::create_dir_all(&cell.directory)?;
    let lock = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(cell.directory.join("cell.lock"))?;
    lock.lock()?;
    let previous = cell.checkpoint()?;
    let mut engine = cell.engine()?;
    let mut start = 0;
    if let Some((record, state)) = previous {
        start = record["step"].as_u64().ok_or("missing update number")? as usize;
        if record["passed"] == true || start == until {
            println!("{record}");
            return Ok(());
        }
        if start > until {
            return Err("cutoff precedes the latest checkpoint".into());
        }
        engine.restore_state(&state)?;
    }

    let train = &cell.task.splits["train"];
    let heldout = &cell.task.splits[&cell.task.selection];
    let per_epoch = train.len().div_ceil(cell.batch);
    let mut random = Random::new(seed + 10_000);
    let mut epoch = Vec::new();
    let mut start_epoch = |epoch: &mut Vec<u32>| {
        epoch.clear();
        epoch.extend(0..train.len() as u32);
        random.shuffle(epoch);
    };
    for _ in 0..start.div_ceil(per_epoch) {
        start_epoch(&mut epoch);
    }
    let mut packed = metrics::Packed::default();
    let started = std::time::Instant::now();
    for step in start..=until {
        if (step != start || start == 0) && (step.is_multiple_of(cell.every) || step == budget) {
            let score = metrics::evaluate(&mut engine, heldout, true)?;
            let correct = score["exact"].as_u64().ok_or("missing exact count")? as usize;
            let passed = correct * 100 >= 99 * heldout.len();
            let result = json!({"identity":cell.identity(),"step":step,"correct":correct,"total":heldout.len(),
                "loss":score["loss"],"passed":passed,"budget":budget,"gpu":engine.cuda.name(),
                "segment_seconds":started.elapsed().as_secs_f64()});
            eprintln!("{result}");
            if passed || step == until {
                return cell.save(&engine, step, result);
            }
        }
        if step.is_multiple_of(per_epoch) {
            start_epoch(&mut epoch);
        }
        let lo = step % per_epoch * cell.batch;
        let indices = &epoch[lo..(lo + cell.batch).min(epoch.len())];
        update(
            &mut engine,
            train,
            &mut packed,
            indices,
            cell.schedule.rate(step),
            step,
        )?;
    }
    unreachable!("the requested final step is always scored")
}

fn update(
    engine: &mut Engine,
    train: &[Example],
    p: &mut metrics::Packed,
    indices: &[u32],
    rate: f32,
    step: usize,
) -> Result<()> {
    p.fill(train, indices);
    engine.forward(&p.tokens, &p.labels, indices.len(), p.length)?;
    engine.step(rate, step)
}

fn replay(name: &str, seed: u32, repeat: usize, out: &Path, variant: Variant) -> Result<()> {
    let cell = Cell::new(name, seed, repeat, out, variant)?;
    let (record, state) = cell.checkpoint()?.ok_or("missing decoder checkpoint")?;
    let mut engine = cell.engine()?;
    engine.restore_state(&state)?;
    let mut splits = serde_json::Map::new();
    for (split, examples) in &cell.task.splits {
        if split != "train" && split != "train_clean" {
            let score = metrics::evaluate(&mut engine, examples, true)?;
            splits.insert(
                split.clone(),
                json!({"correct":score["exact"],"total":examples.len(),"loss":score["loss"]}),
            );
        }
    }
    let primary = &splits[&cell.task.selection];
    if primary["correct"] != record["correct"] || primary["total"] != record["total"] {
        return Err("independent replay disagreed with saved primary score".into());
    }
    println!(
        "{}",
        json!({"identity":cell.identity(),"step":record["step"],"state_sha256":record["state_sha256"],"splits":splits})
    );
    Ok(())
}

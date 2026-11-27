//! Frozen GPT stress cases and schedule for the Argand full-batch runs.
use super::{Example, Result, Task, Value, fs, hash, json, load, publish, suite_hash, task_config};
use crate::{
    Config, KERNITE_VERSION,
    argand::{Engine, MODEL, PENALTY},
    argand_metrics as metrics,
    schedule::Schedule,
};
use std::path::{Path, PathBuf};

pub fn command(op: &str, args: &[String]) -> Result<()> {
    match (op, args) {
        ("argand-check", []) => crate::argand::check(),
        ("argand-run", [name, seed, repeat, out, cutoff @ ..]) if cutoff.len() <= 1 => run(
            name,
            seed.parse()?,
            repeat.parse()?,
            Path::new(out),
            cutoff.first().map(|step| step.parse()).transpose()?,
        ),
        ("argand-replay", [name, seed, repeat, out]) => {
            replay(name, seed.parse()?, repeat.parse()?, Path::new(out))
        }
        _ => Err("usage: argand-check | argand-run CASE SEED REPEAT RUN_DIR [LAST_OBSERVATION] | argand-replay CASE SEED REPEAT RUN_DIR".into()),
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
}

impl Cell {
    fn new(name: &str, seed: u32, repeat: usize, out: &Path) -> Result<Self> {
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
        })
    }

    fn engine(&self) -> Result<Engine> {
        let mut engine = Engine::new(self.shape.clone(), self.repeat, self.seed)?;
        engine.config.max_batch = task_config(&self.row, &self.task, false)?.max_batch;
        Ok(engine)
    }

    fn identity(&self) -> Value {
        json!({"model":MODEL,"repeat":self.repeat,"seed":self.seed,"case":self.row["name"],
            "tier":self.row["tier"],"suite_sha256":suite_hash(&self.row),"kernite":KERNITE_VERSION,
            "penalty":PENALTY.to_string()})
    }

    fn state_file(&self, step: usize) -> PathBuf {
        self.directory.join(format!("state-{step}.f32"))
    }

    fn checkpoint(&self) -> Result<Option<(Value, Vec<f32>)>> {
        let path = self.directory.join("result.json");
        if !path.try_exists()? {
            return Ok(None);
        }
        let record: Value = serde_json::from_slice(&fs::read(path)?)?;
        let step = record["step"].as_u64().ok_or("missing update number")? as usize;
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

fn run(name: &str, seed: u32, repeat: usize, out: &Path, cutoff: Option<usize>) -> Result<()> {
    let cell = Cell::new(name, seed, repeat, out)?;
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
    let mut packed = metrics::Packed::default();
    let full = FullBatch::new(&engine, train, &mut packed, cell.batch)?;
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
        full.update(&mut engine, train, &mut packed, cell.schedule.rate(step))?;
    }
    unreachable!("the requested final step is always scored")
}

/// The training split in order, as microbatches of the suite's batch size.
struct FullBatch {
    order: Vec<u32>,
    batch: usize,
    labeled: usize,
}

impl FullBatch {
    fn new(
        engine: &Engine,
        train: &[Example],
        p: &mut metrics::Packed,
        batch: usize,
    ) -> Result<Self> {
        let order: Vec<u32> = (0..train.len() as u32).collect();
        let labeled = order
            .chunks(batch)
            .map(|part| {
                p.fill(train, part);
                engine.labeled(&p.labels)
            })
            .sum();
        if labeled == 0 {
            return Err("full-batch training needs a labeled training position".into());
        }
        Ok(Self {
            order,
            batch,
            labeled,
        })
    }

    /// One update of the Lean `fullGradientMomentumRun`: the microbatches
    /// accumulate the gradient of the mean loss over all labeled training
    /// positions, then the penalty and the clipped momentum step follow.
    fn update(
        &self,
        engine: &mut Engine,
        train: &[Example],
        p: &mut metrics::Packed,
        rate: f32,
    ) -> Result<()> {
        engine.zero_gradient()?;
        for part in self.order.chunks(self.batch) {
            p.fill(train, part);
            engine.accumulate(&p.tokens, &p.labels, part.len(), p.length, self.labeled)?;
        }
        engine.full_step(rate)
    }
}

fn replay(name: &str, seed: u32, repeat: usize, out: &Path) -> Result<()> {
    let cell = Cell::new(name, seed, repeat, out)?;
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

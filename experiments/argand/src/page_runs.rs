//! Optimizer comparisons with identical initialization, shuffled epochs and scoring.
use crate::{
    KERNITE_VERSION, Result,
    data::{Example, Task},
    division,
    engine::{
        Engine,
        adamw::{AdamW, RatePlan, wsd},
        page::{AdaptivePage, Observations, PageRule},
    },
    metrics,
    random::Random,
    schedule::Schedule,
    suite::{self, Shape, load_weights, publish, save_weights},
};
use serde_json::{Value, json};
use std::{fs, io::Write, path::Path, time::Instant};

#[derive(Clone, Copy)]
pub(crate) enum Rule {
    Page(PageRule),
    AdamW(RatePlan),
    AdamWFirstNoise { seed: u32 },
}

impl Rule {
    pub fn label(self) -> &'static str {
        match self {
            Self::Page(correction) => correction.label(),
            Self::AdamW(RatePlan::Wsd) => "adamw-wsd",
            Self::AdamW(RatePlan::Geometric { .. }) => "adamw-wsd-geometric-tail",
            Self::AdamW(RatePlan::Restart { .. }) => "adamw-wsd-restart",
            Self::AdamWFirstNoise { .. } => "adamw-wsd-first-noise",
        }
    }
}

pub(crate) enum Optimizer {
    Page(AdaptivePage),
    AdamW(AdamW),
}

impl Optimizer {
    pub fn new(
        engine: &mut Engine,
        rule: Rule,
        batch: Observations<'_>,
        sampler: u32,
        schedule: Schedule,
    ) -> Result<Self> {
        Ok(match rule {
            Rule::Page(correction) => {
                Self::Page(AdaptivePage::new(engine, batch, sampler, correction)?)
            }
            Rule::AdamW(plan) => Self::AdamW(AdamW::new(engine, schedule, plan)?),
            Rule::AdamWFirstNoise { seed } => {
                Self::AdamW(AdamW::new(engine, schedule, RatePlan::Wsd)?.with_first_noise(seed))
            }
        })
    }

    pub fn step(
        &mut self,
        engine: &mut Engine,
        batch: Observations<'_>,
        changed: bool,
    ) -> Result<bool> {
        match self {
            Self::Page(page) => page.step(engine, batch, changed),
            Self::AdamW(adamw) => {
                adamw.step(engine, batch)?;
                Ok(true)
            }
        }
    }

    pub fn metrics(&self) -> Value {
        match self {
            Self::Page(page) => page.metrics(),
            Self::AdamW(adamw) => adamw.metrics(),
        }
    }
}

/// The least true-token gap over all labeled prefixes. Strict correctness on
/// every true prefix also implies exact greedy rollout, by induction.
fn strict(engine: &mut Engine, examples: &[Example]) -> Result<Value> {
    let mut packed = metrics::Packed::default();
    let mut gap = f64::INFINITY;
    let (mut correct, mut ties, mut labels) = (0, 0, 0);
    let order: Vec<u32> = (0..examples.len() as u32).collect();
    for chunk in order.chunks(engine.config.max_batch) {
        packed.fill(examples, chunk);
        for row in crate::margins::rows(engine, &packed)? {
            gap = gap.min(row.gap);
            ties += row.ties;
            labels += row.labels;
            correct += usize::from(row.gap > 0.0);
        }
    }
    Ok(
        json!({"strict_correct":correct,"total":examples.len(),"labeled_tokens":labels,
        "ties":ties,"min_gap":gap,"all_strict":correct==examples.len()}),
    )
}

pub struct Job {
    row: Value,
    task: Task,
    shape: Shape,
    batch: usize,
    every: usize,
    pub until: usize,
    ordered_gradients: bool,
}

impl Job {
    pub fn new(name: &str, cap: usize) -> Result<Self> {
        let (row, task) = if name == division::NAME {
            division::load()?
        } else {
            suite::load(name)?
        };
        let shape = suite::shape(&row, &task)?;
        let get = |k: &str| {
            row["case"][k]
                .as_u64()
                .map(|n| n as usize)
                .ok_or("missing case setting")
        };
        let (every, batch, budget) = (get("observe")?, get("batch")?, get("max_updates")?);
        let until = if name == division::NAME && cap > budget {
            cap
        } else if cap >= budget {
            budget
        } else {
            cap / every * every
        };
        if until == 0 {
            return Err("cutoff precedes the first positive observation".into());
        }
        Ok(Self {
            row,
            task,
            shape,
            batch,
            every,
            until,
            ordered_gradients: false,
        })
    }

    pub fn with_ordered_gradients(mut self, ordered: bool) -> Self {
        self.ordered_gradients = ordered;
        self
    }

    fn verify(
        &self,
        dir: &Path,
        record: &Value,
        engine: &mut Engine,
        weights: &str,
        replay: &str,
    ) -> Result<()> {
        load_weights(dir, weights, &record["weights_sha256"], engine)?;
        let mut splits = serde_json::Map::new();
        for (split, examples) in &self.task.splits {
            if split == "train" || split == "train_clean" {
                continue;
            }
            let score = metrics::evaluate(engine, examples, true)?;
            splits.insert(
                split.clone(),
                json!({"correct":score["exact"],"total":examples.len(),"loss":score["loss"]}),
            );
        }
        let primary = &splits[&self.task.selection];
        if primary["correct"] != record["correct"]
            || primary["total"] != record["total"]
            || primary["loss"] != record["loss"]
            || strict(engine, &self.task.splits[&self.task.selection])? != record["strict"]
        {
            return Err("independent checkpoint replay disagrees with the saved scores".into());
        }
        publish(
            dir,
            replay,
            &serde_json::to_vec_pretty(&json!({"identity":record["identity"],
            "weights_sha256":record["weights_sha256"],"step":record["step"],"strict":record["strict"],"splits":splits}))?,
        )
    }

    /// With a fixed horizon, intermediate scores never stop or change training.
    pub fn run(
        &self,
        dir: &Path,
        seed: u32,
        sampler: u32,
        rule: Rule,
        fixed: bool,
    ) -> Result<Value> {
        fs::create_dir_all(dir)?;
        let lock = fs::File::options()
            .create(true)
            .truncate(false)
            .write(true)
            .open(dir.join("cell.lock"))?;
        lock.lock()?;
        let mut identity = json!({"case":self.row["name"],"tier":self.row["tier"],"model":"argand-skip-write-tied-score-init-one",
            "seed":seed,"order_seed":seed.wrapping_add(10000),"split_sha256":self.row["split_sha256"],
            "suite_sha256":suite::suite_hash(),"kernite":KERNITE_VERSION,"repeat":1,"budget":self.until,
            "fixed_horizon":fixed,"width":self.shape.width,"heads":self.shape.heads,"layers":self.shape.layers,
            "batch":self.batch,"observe":self.every,"vocab":self.shape.vocab});
        if self.ordered_gradients {
            identity["gradient_reduction"] = "ordered-f32-v1".into();
        }
        let geometric_after = match rule {
            Rule::AdamW(RatePlan::Geometric { budget }) => Some(budget),
            _ => None,
        };
        if geometric_after.is_some_and(|b| b < 10 || b > self.until) {
            return Err(
                "geometric continuation requires a WSD prefix between 10 and the cutoff".into(),
            );
        }
        let budget = if let Some(budget) = geometric_after {
            budget
        } else if self.row["name"] == division::NAME
            || matches!(rule, Rule::AdamW(RatePlan::Restart { .. }))
        {
            self.until
        } else {
            self.row["case"]["max_updates"]
                .as_u64()
                .ok_or("missing WSD budget")? as usize
        };
        let schedule = wsd(budget);
        match rule {
            Rule::Page(correction) => {
                identity["optimizer"] = "batch-reference-adaptive-page-v1".into();
                identity["correction"] = correction.label().into();
                identity["page_seed"] = sampler.into();
            }
            Rule::AdamW(plan) => {
                identity["optimizer"] = rule.label().into();
                identity["adamw"] = plan.settings(schedule);
                if let RatePlan::Restart {
                    stock_runs,
                    trial,
                    first_budget,
                } = plan
                {
                    identity["restart"] = json!({"stock_runs":stock_runs,"trial":trial,"first_budget":first_budget,"rate_scale":plan.scale(budget)});
                }
            }
            Rule::AdamWFirstNoise { seed } => {
                identity["optimizer"] = rule.label().into();
                identity["adamw"] = RatePlan::Wsd.settings(schedule);
                identity["gaussian_weight_noise"] = json!({"after_update":1,"seed":seed,
                    "stddev":crate::engine::adamw::FIRST_NOISE_STDDEV});
            }
        }
        let mut engine = Engine::new(self.shape.clone(), 1, seed)?;
        if dir.join("result.json").try_exists()? {
            let record: Value = serde_json::from_slice(&fs::read(dir.join("result.json"))?)?;
            if record["identity"] != identity {
                return Err("incompatible optimizer checkpoint".into());
            }
            self.verify(dir, &record, &mut engine, "weights.f32", "replay.json")?;
            return Ok(record);
        }
        if self.ordered_gradients {
            engine.order_shared_gradients()?;
        }
        let train = &self.task.splits["train"];
        let hidden = &self.task.splits[&self.task.selection];
        let mut rng = Random::new(seed.wrapping_add(10000));
        let mut epoch: Vec<u32> = (0..u32::try_from(train.len())?).collect();
        let batches = epoch.len().div_ceil(self.batch);
        let mut packed = metrics::Packed::default();
        let mut optimizer: Option<Optimizer> = None;
        let mut skipped = 0;
        let mut first_strict = None;
        let mut log = fs::File::create(dir.join("observations.jsonl"))?;
        let start = Instant::now();
        let division = self.row["name"] == division::NAME;
        for step in 0..=self.until {
            if step.is_multiple_of(self.every)
                || step == self.until
                || Some(step) == geometric_after
            {
                let score = metrics::evaluate(&mut engine, hidden, true)?;
                let correct = score["exact"].as_u64().ok_or("missing exact score")? as usize;
                let passed = correct * 100 >= 99 * hidden.len();
                let finish = step == self.until || (!fixed && passed);
                let train_correct = if division {
                    Some(metrics::evaluate(&mut engine, train, false)?["exact"].clone())
                } else {
                    None
                };
                let mut record = json!({"identity":identity,"step":step,"trained":train_correct,"training_rows":train.len(),
                    "correct":correct,"total":hidden.len(),"passed":passed,"loss":score["loss"],
                    "segment_seconds":start.elapsed().as_secs_f64(),"gpu":engine.cuda.name(),"skipped_batches":skipped,
                    "optimizer":optimizer.as_ref().map(Optimizer::metrics)});
                let check_first = fixed
                    && division
                    && first_strict.is_none()
                    && step > 0
                    && correct == hidden.len();
                if finish || check_first {
                    record["strict"] = strict(&mut engine, hidden)?;
                }
                if check_first && record["strict"]["all_strict"] == true {
                    first_strict = Some(step);
                    let (_, digest) = save_weights(dir, "first-strict.f32", &engine)?;
                    record["weights_sha256"] = digest.into();
                    publish(
                        dir,
                        "first-strict.json",
                        &serde_json::to_vec_pretty(&record)?,
                    )?;
                    let mut check = Engine::new(self.shape.clone(), 1, seed)?;
                    self.verify(
                        dir,
                        &record,
                        &mut check,
                        "first-strict.f32",
                        "first-strict-replay.json",
                    )?;
                }
                if finish {
                    if division {
                        record["first_strict_step"] = first_strict.into();
                    }
                    let (_, digest) = save_weights(dir, "weights.f32", &engine)?;
                    record["weights_sha256"] = digest.into();
                }
                writeln!(log, "{record}")?;
                log.flush()?;
                if finish {
                    publish(dir, "result.json", &serde_json::to_vec_pretty(&record)?)?;
                    drop(engine);
                    let mut copy = Engine::new(self.shape.clone(), 1, seed)?;
                    self.verify(dir, &record, &mut copy, "weights.f32", "replay.json")?;
                    return Ok(record);
                }
            }
            let offset = step % batches * self.batch;
            if offset == 0 {
                for (i, v) in epoch.iter_mut().enumerate() {
                    *v = i as u32;
                }
                rng.shuffle(&mut epoch);
            }
            packed.fill(
                train,
                &epoch[offset..(offset + self.batch).min(epoch.len())],
            );
            let reference = Observations::from(&packed);
            if optimizer.is_none() {
                optimizer = Some(Optimizer::new(
                    &mut engine,
                    rule,
                    reference,
                    sampler,
                    schedule,
                )?);
            }
            if !optimizer
                .as_mut()
                .unwrap()
                .step(&mut engine, reference, step > 0)?
            {
                skipped += 1;
            }
        }
        unreachable!("the terminal checkpoint is always observed")
    }
}

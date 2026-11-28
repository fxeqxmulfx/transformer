//! Training, scoring and replay of one Argand cell of the frozen GPT tiers.
use crate::{
    KERNITE_VERSION, Result, bench,
    data::{Example, Task},
    engine::{self, Engine, MODEL, PENALTY, adamw::RatePlan},
    hash, metrics,
    page_runs::Stop,
    schedule::Schedule,
    suite::{self, Shape, publish, suite_hash},
};
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
};

const USAGE: &str = concat!(
    "usage: argand check | peak | bench CASE UPDATES | run CASE SEED REPEAT RUN_DIR [LAST_OBSERVATION] | ",
    "replay CASE SEED REPEAT RUN_DIR | (sum8-page | sum8-replay)[-sqrt|-retry] RUN_DIR [UPDATES] | ",
    "sum8-adamw[-first-noise[-until-perfect]][-replay] RUN_DIR [UPDATES] | (adamw-run | page-run[-sqrt|-retry]) CASE SEED RUN_DIR CAP [--ordered-gradients] | ",
    "sum8-adamw-rank-one[-margin][-replay] RUN_DIR UPDATES [DRAW_INDEX] | sum8-rectangle-certificate CELL | ",
    "adamw-restart-run CASE SEED RUN_DIR CAP STOCK_RUNS TRIAL FIRST_BUDGET [--ordered-gradients] | ",
    "adamw-until-perfect CASE SEED RUN_DIR CAP | adamw-first-noise-(run|screen|until-perfect) CASE SEED NOISE_SEED RUN_DIR CAP [--ordered-gradients] | ",
    "(adamw-scaled-until-perfect | softsign-until-perfect | muon-until-perfect | equal-moment-until-perfect | uncorrected-adam-until-perfect | normalized-adamw-until-perfect) CASE SEED RUN_DIR CAP RATE_MULTIPLIER [WSD_BUDGET] [--ordered-gradients] | ",
    "(softsign-checkpoint-proposals | rmsprop-checkpoint-proposals | equal-moment-checkpoint-proposals | uncorrected-adam-checkpoint-proposals | normalized-adamw-checkpoint-proposals) CASE SEED PROPOSAL_SEED OUT CAP FIRST_PROPOSAL_UPDATE RATE_MULTIPLIER [WSD_BUDGET [WEIGHT_DECAY [SOFTSIGN_ROW_NORM]]] | ",
    "adamw-checkpoint-proposals CASE SEED PROPOSAL_SEED OUT CAP FIRST_PROPOSAL_UPDATE [RATE_MULTIPLIER [WSD_BUDGET]] | sum8-checkpoint-proposals STOCK_CELL PROPOSAL_SEED OUT FIRST_PROPOSAL_UPDATE | ",
    "adamw-tail-run CASE SEED RUN_DIR WSD_BUDGET CAP [--ordered-gradients] | adamw-scaled-run CASE SEED RUN_DIR CAP RATE_MULTIPLIER WSD_BUDGET [--ordered-gradients] | (adamw-confirm | page-confirm[-sqrt|-retry]) CASE RUN_DIR CAP | ",
    "division-certificate CELL (terminal|first-strict) | division-proposal-mass CELL SEED_FILE DRAWS | division-recorded-lean CELL (terminal|first-strict) [--hidden] | division-project-rows CELL (terminal|first-strict) RADIUS | division-relabel CELL | division-mask-lean"
);

fn training_rule(op: &str) -> crate::page_runs::Rule {
    if op.starts_with("adamw-") {
        crate::page_runs::Rule::AdamW(RatePlan::Wsd)
    } else {
        crate::page_runs::Rule::Page(correction(op))
    }
}

fn rate_multiplier(scale: &str) -> Result<f64> {
    let scale: f64 = scale.parse()?;
    if !scale.is_finite()
        || scale <= 0.0
        || crate::schedule::PEAK * scale >= f64::from(f32::MAX)
        || (1e-8 * scale) as f32 == 0.0
    {
        return Err("expected a positive finite WSD rate multiplier".into());
    }
    Ok(scale)
}

pub fn command(args: &[String]) -> Result<()> {
    let Some((op, args)) = args.split_first() else {
        return Err(USAGE.into());
    };
    let (args, ordered_gradients) = match args {
        [rest @ .., flag]
            if flag == "--ordered-gradients"
                && matches!(
                    op.as_str(),
                    "adamw-run"
                        | "adamw-until-perfect"
                        | "adamw-scaled-until-perfect"
                        | "softsign-until-perfect"
                        | "muon-until-perfect"
                        | "rmsprop-checkpoint-proposals"
                        | "equal-moment-checkpoint-proposals"
                        | "uncorrected-adam-checkpoint-proposals"
                        | "normalized-adamw-checkpoint-proposals"
                        | "equal-moment-until-perfect"
                        | "uncorrected-adam-until-perfect"
                        | "normalized-adamw-until-perfect"
                        | "softsign-checkpoint-proposals"
                        | "adamw-checkpoint-proposals"
                        | "adamw-first-noise-until-perfect"
                        | "adamw-first-noise-run"
                        | "adamw-first-noise-screen"
                        | "adamw-restart-run"
                        | "adamw-tail-run"
                        | "adamw-scaled-run"
                        | "page-run"
                        | "page-run-sqrt"
                        | "page-run-retry"
                ) =>
        {
            (rest, true)
        }
        _ => (args, false),
    };
    match (op.as_str(), args) {
        (
            "adamw-scaled-until-perfect"
            | "softsign-until-perfect"
            | "muon-until-perfect"
            | "equal-moment-until-perfect"
            | "uncorrected-adam-until-perfect"
            | "normalized-adamw-until-perfect",
            [name, seed, out, cap, scale, budget @ ..],
        ) if budget.len() <= 1 => {
            let scale = rate_multiplier(scale)?;
            let seed: u32 = seed.parse()?;
            let mut job = crate::page_runs::Job::new(name, cap.parse()?)?
                .with_ordered_gradients(ordered_gradients);
            if let Some(budget) = budget.first() {
                job = job.with_schedule_budget(budget.parse()?)?;
            }
            let dir = Path::new(out).join(name).join(format!("seed{seed}"));
            let rule = match op.as_str() {
                "softsign-until-perfect" => crate::page_runs::Rule::Softsign {
                    scale,
                    decay: 0.1,
                    row_norm: None,
                },
                "muon-until-perfect" => crate::page_runs::Rule::Muon(scale),
                "equal-moment-until-perfect"
                | "uncorrected-adam-until-perfect"
                | "normalized-adamw-until-perfect" => {
                    use crate::engine::rmsprop::MomentRates;
                    crate::page_runs::Rule::MomentAdam {
                        scale,
                        decay: 0.1,
                        rates: match op.as_str() {
                            "equal-moment-until-perfect" => MomentRates::Equal,
                            "uncorrected-adam-until-perfect" => MomentRates::HalfSecondGain,
                            _ => MomentRates::NORMALIZED,
                        },
                    }
                }
                _ => crate::page_runs::Rule::AdamW(RatePlan::Scaled(scale)),
            };
            println!("{}", job.run(&dir, seed, 0, rule, Stop::Perfect)?);
            Ok(())
        }
        (
            "softsign-checkpoint-proposals"
            | "rmsprop-checkpoint-proposals"
            | "equal-moment-checkpoint-proposals"
            | "uncorrected-adam-checkpoint-proposals"
            | "normalized-adamw-checkpoint-proposals",
            [name, seed, noise_seed, out, cap, first, scale, budget @ ..],
        ) if budget.len() <= 2 || (op == "softsign-checkpoint-proposals" && budget.len() == 3) => {
            let seed = seed.parse()?;
            let proposal = noise_seed.parse()?;
            let first = first.parse()?;
            let cap = cap.parse()?;
            let row_norm: Option<f64> = budget.get(2).map(|s| s.parse()).transpose()?;
            if row_norm
                .is_some_and(|r| !r.is_finite() || (r as f32) <= 0.0 || r > f64::from(f32::MAX))
            {
                return Err("expected a positive finite projection row radius".into());
            }
            let decay: f64 = budget.get(1).map(|s| s.parse()).transpose()?.unwrap_or(0.1);
            if !decay.is_finite() || decay < 0.0 || decay > f64::from(f32::MAX) {
                return Err("expected a finite nonnegative weight decay".into());
            }
            let budget = budget
                .first()
                .map(|s| s.parse())
                .transpose()?
                .unwrap_or(cap);
            let job = crate::page_runs::Job::new(name, cap)?
                .with_schedule_budget(budget)?
                .with_ordered_gradients(ordered_gradients);
            let dir = Path::new(out)
                .join(name)
                .join(format!("seed{seed}-proposal{proposal}-from{first}"));
            let scale = rate_multiplier(scale)?;
            let rule = match op.as_str() {
                "rmsprop-checkpoint-proposals" => crate::page_runs::Rule::RmsProp { scale, decay },
                "softsign-checkpoint-proposals" => crate::page_runs::Rule::Softsign {
                    scale,
                    decay,
                    row_norm,
                },
                _ => crate::page_runs::Rule::MomentAdam {
                    scale,
                    decay,
                    rates: match op.as_str() {
                        "equal-moment-checkpoint-proposals" => {
                            crate::engine::rmsprop::MomentRates::Equal
                        }
                        "uncorrected-adam-checkpoint-proposals" => {
                            crate::engine::rmsprop::MomentRates::HalfSecondGain
                        }
                        _ => crate::engine::rmsprop::MomentRates::NORMALIZED,
                    },
                },
            };
            let stop = Stop::CheckpointProposal {
                seed: proposal,
                first,
            };
            println!("{}", job.run(&dir, seed, 0, rule, stop)?);
            Ok(())
        }
        (
            "adamw-checkpoint-proposals",
            [name, seed, noise_seed, out, cap, first, settings @ ..],
        ) if settings.len() <= 2 => {
            let mut job = crate::page_runs::Job::new(name, cap.parse()?)?
                .with_ordered_gradients(ordered_gradients);
            let plan = match settings.first() {
                Some(scale) => RatePlan::Scaled(rate_multiplier(scale)?),
                None => RatePlan::Wsd,
            };
            if let Some(budget) = settings.get(1) {
                job = job.with_schedule_budget(budget.parse()?)?;
            }
            let seed = seed.parse()?;
            let proposal = noise_seed.parse()?;
            let first = first.parse()?;
            let dir = Path::new(out)
                .join(name)
                .join(format!("seed{seed}-proposal{proposal}-from{first}"));
            println!(
                "{}",
                job.run(
                    &dir,
                    seed,
                    0,
                    crate::page_runs::Rule::AdamW(plan),
                    Stop::CheckpointProposal {
                        seed: proposal,
                        first
                    }
                )?
            );
            Ok(())
        }
        ("sum8-checkpoint-proposals", [cell, seed, out, first]) => {
            println!(
                "{}",
                crate::sum8::checkpoint_proposals(
                    Path::new(cell),
                    seed.parse()?,
                    Path::new(out),
                    first.parse()?
                )?
            );
            Ok(())
        }
        ("division-mask-lean", []) => {
            print!("{}", crate::division_certificate::mask::lean_source()?);
            Ok(())
        }
        ("division-proposal-mass", [cell, seeds, draws]) => {
            println!(
                "{}",
                crate::division_certificate::proposal_mass::measure(
                    Path::new(cell),
                    Path::new(seeds),
                    draws.parse()?
                )?
            );
            Ok(())
        }
        ("division-certificate", [cell, kind]) if kind == "terminal" || kind == "first-strict" => {
            println!(
                "{}",
                crate::division_certificate::check(Path::new(cell), kind == "first-strict")?
            );
            Ok(())
        }
        ("division-relabel", [cell]) => {
            println!(
                "{}",
                crate::division_certificate::relabel::check(Path::new(cell))?
            );
            Ok(())
        }
        ("division-recorded-lean", [cell, kind, scope @ ..])
            if (kind == "terminal" || kind == "first-strict")
                && (scope.is_empty() || matches!(scope, [flag] if flag == "--hidden")) =>
        {
            println!(
                "{}",
                crate::division_certificate::recorded_scores::export(
                    Path::new(cell),
                    kind == "first-strict",
                    !scope.is_empty()
                )?
            );
            Ok(())
        }
        ("division-project-rows", [cell, kind, radius])
            if kind == "terminal" || kind == "first-strict" =>
        {
            println!(
                "{}",
                crate::division_certificate::row_projection::check(
                    Path::new(cell),
                    kind == "first-strict",
                    radius.parse()?
                )?
            );
            Ok(())
        }
        ("adamw-scaled-run", [name, seed, out, cap, scale, budget]) => {
            let scale = rate_multiplier(scale)?;
            let seed: u32 = seed.parse()?;
            let job = crate::page_runs::Job::new(name, cap.parse()?)?
                .with_schedule_budget(budget.parse()?)?
                .with_ordered_gradients(ordered_gradients);
            let rule = crate::page_runs::Rule::AdamW(RatePlan::Scaled(scale));
            let dir = Path::new(out).join(name).join(format!("seed{seed}"));
            println!("{}", job.run(&dir, seed, 0, rule, Stop::Persistence)?);
            Ok(())
        }
        ("adamw-tail-run", [name, seed, out, budget, cap]) => {
            let seed: u32 = seed.parse()?;
            let job = crate::page_runs::Job::new(name, cap.parse()?)?
                .with_ordered_gradients(ordered_gradients);
            let rule = crate::page_runs::Rule::AdamW(RatePlan::Geometric {
                budget: budget.parse()?,
            });
            let dir = Path::new(out).join(name).join(format!("seed{seed}"));
            println!("{}", job.run(&dir, seed, 0, rule, Stop::Horizon)?);
            Ok(())
        }
        ("adamw-restart-run", [name, seed, out, cap, stock, index, first]) => {
            let job = crate::page_runs::Job::new(name, cap.parse()?)?
                .with_ordered_gradients(ordered_gradients);
            let trial: usize = index.parse()?;
            let first_budget: usize = first.parse()?;
            if trial == 0 && first_budget != job.until {
                return Err(
                    "first restart budget must equal the first trial's effective horizon".into(),
                );
            }
            let rule = crate::page_runs::Rule::AdamW(RatePlan::Restart {
                stock_runs: stock.parse()?,
                trial,
                first_budget,
            });
            let dir = Path::new(out).join(name).join(format!("draw-{trial}"));
            println!("{}", job.run(&dir, seed.parse()?, 0, rule, Stop::Horizon)?);
            Ok(())
        }
        (
            "adamw-first-noise-run"
            | "adamw-first-noise-screen"
            | "adamw-first-noise-until-perfect",
            [name, seed, noise_seed, out, cap],
        ) => {
            let seed: u32 = seed.parse()?;
            let noise_seed: u32 = noise_seed.parse()?;
            let job = crate::page_runs::Job::new(name, cap.parse()?)?
                .with_ordered_gradients(ordered_gradients);
            let dir = Path::new(out)
                .join(name)
                .join(format!("seed{seed}-noise{noise_seed}"));
            let rule = crate::page_runs::Rule::AdamWFirstNoise { seed: noise_seed };
            let stop = match op.as_str() {
                "adamw-first-noise-run" => Stop::Horizon,
                "adamw-first-noise-screen" => Stop::Pass99,
                _ => Stop::Perfect,
            };
            println!("{}", job.run(&dir, seed, 0, rule, stop)?);
            Ok(())
        }
        ("check", []) => engine::check(),
        ("peak", []) => bench::peak(),
        ("sum8-spectrum", [out]) => crate::sum8_spectrum::run(Path::new(out)),
        ("sum8-rectangle-certificate", [cell]) => {
            println!("{}", crate::sum8::audit_rectangles(Path::new(cell))?);
            Ok(())
        }
        (
            "sum8-adamw-rank-one"
            | "sum8-adamw-rank-one-replay"
            | "sum8-adamw-rank-one-margin"
            | "sum8-adamw-rank-one-margin-replay",
            [out, cap, draw @ ..],
        ) if draw.len() <= 1 => crate::sum8::run(
            Path::new(out),
            Some(cap.parse()?),
            op.ends_with("-replay"),
            crate::sum8::Method::AdamWRankOne {
                margin: op.contains("-margin"),
            },
            draw.first().map(|v| v.parse()).transpose()?,
        ),
        ("page-and-audit", [cell]) => {
            println!("{}", crate::and_audit::run(Path::new(cell))?);
            Ok(())
        }
        (
            "adamw-run" | "adamw-until-perfect" | "page-run" | "page-run-sqrt" | "page-run-retry",
            [name, seed, out, cap],
        ) => {
            let seed: u32 = seed.parse()?;
            let job = crate::page_runs::Job::new(name, cap.parse()?)?
                .with_ordered_gradients(ordered_gradients);
            let dir = Path::new(out).join(name).join(format!("seed{seed}"));
            let record = job.run(
                &dir,
                seed,
                seed.wrapping_add(30000),
                training_rule(op),
                if op == "adamw-until-perfect" {
                    Stop::Perfect
                } else if name == crate::division::NAME {
                    Stop::Horizon
                } else {
                    Stop::Pass99
                },
            )?;
            println!("{record}");
            Ok(())
        }
        (
            "adamw-confirm" | "adamw-confirm-plan" | "page-confirm" | "page-confirm-plan"
            | "page-confirm-sqrt" | "page-confirm-retry",
            [name, out, cap],
        ) => crate::page_confirmation::run(
            name,
            Path::new(out),
            cap.parse()?,
            training_rule(op),
            op.ends_with("-plan"),
        ),
        (
            "sum8-adamw"
            | "sum8-adamw-replay"
            | "sum8-adamw-first-noise"
            | "sum8-adamw-first-noise-replay"
            | "sum8-adamw-first-noise-until-perfect"
            | "sum8-adamw-first-noise-until-perfect-replay"
            | "sum8-page"
            | "sum8-replay"
            | "sum8-page-sqrt"
            | "sum8-replay-sqrt"
            | "sum8-page-retry"
            | "sum8-replay-retry",
            [out, steps @ ..],
        ) if steps.len() <= 1 => crate::sum8::run(
            Path::new(out),
            steps.first().map(|s| s.parse()).transpose()?,
            op.contains("replay"),
            if op.starts_with("sum8-adamw-first-noise") {
                crate::sum8::Method::AdamWFirstNoise {
                    until_perfect: op.contains("until-perfect"),
                }
            } else if op.starts_with("sum8-adamw") {
                crate::sum8::Method::AdamW
            } else {
                crate::sum8::Method::Page(correction(op))
            },
            None,
        ),
        ("bench", [name, updates]) => bench::bench(name, updates.parse()?),
        ("run", [name, seed, repeat, out, cutoff @ ..]) if cutoff.len() <= 1 => run(
            name,
            seed.parse()?,
            repeat.parse()?,
            Path::new(out),
            cutoff.first().map(|step| step.parse()).transpose()?,
        ),
        ("replay", [name, seed, repeat, out]) => {
            replay(name, seed.parse()?, repeat.parse()?, Path::new(out))
        }
        _ => Err(USAGE.into()),
    }
}

fn correction(op: &str) -> engine::page::PageRule {
    match op.rsplit('-').next() {
        Some("sqrt") => engine::page::PageRule::SquareRoot,
        Some("retry") => engine::page::PageRule::Retry,
        _ => engine::page::PageRule::Single,
    }
}

pub(crate) struct Cell {
    row: Value,
    pub task: Task,
    pub shape: Shape,
    pub batch: usize,
    every: usize,
    pub schedule: Schedule,
    directory: PathBuf,
    repeat: usize,
    seed: u32,
}

impl Cell {
    pub fn new(name: &str, seed: u32, repeat: usize, out: &Path) -> Result<Self> {
        if seed > 2 || repeat == 0 {
            return Err("expected seed 0 through 2 and a positive repeat count".into());
        }
        let (row, task) = suite::load(name)?;
        let number = |k: &str| {
            row["case"][k]
                .as_u64()
                .ok_or("missing case setting")
                .map(|n| n as usize)
        };
        let shape = suite::shape(&row, &task)?;
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

    pub fn engine(&self) -> Result<Engine> {
        Engine::new(self.shape.clone(), self.repeat, self.seed)
    }

    fn identity(&self) -> Value {
        json!({"model":MODEL,"repeat":self.repeat,"seed":self.seed,"case":self.row["name"],
            "tier":self.row["tier"],"suite_sha256":suite_hash(),"kernite":KERNITE_VERSION,
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
        let state = suite::read_floats(&self.state_file(step), &record["state_sha256"])?;
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
pub(crate) struct FullBatch {
    order: Vec<u32>,
    batch: usize,
    labeled: usize,
}

impl FullBatch {
    pub fn new(
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
    pub fn update(
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

    /// Accumulates one full-batch gradient, as `update` does without its
    /// step, and counts its matrix-product operations, padded positions and
    /// microbatches.
    pub fn count(
        &self,
        engine: &mut Engine,
        train: &[Example],
        p: &mut metrics::Packed,
    ) -> Result<(f64, usize, usize)> {
        engine.zero_gradient()?;
        let (mut flops, mut positions, mut parts) = (0.0, 0, 0);
        for part in self.order.chunks(self.batch) {
            p.fill(train, part);
            engine.accumulate(&p.tokens, &p.labels, part.len(), p.length, self.labeled)?;
            flops += engine.product_flops();
            positions += p.tokens.len();
            parts += 1;
        }
        Ok((flops, positions, parts))
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

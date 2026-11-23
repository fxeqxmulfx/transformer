//! Train one task-scaled Inkling architecture on the original 35 GPT tasks.
mod depth;
mod engine;
mod optimizer;
#[cfg(test)]
mod tests;
mod text;
use minibatch::{Result, hash};
use minibatch::{arithmetic_data, blake, pcg, random};
use random::Random;
#[path = "../../convex_stress/src/vendor/data.rs"]
pub mod data;
#[path = "../../../.lake/reproduction/gpt/research/complex_geometry_suite/native/src/metrics.rs"]
mod metrics;
#[path = "../../../.lake/reproduction/gpt/research/main_model/wsd_suite_native/src/schedule.rs"]
mod schedule;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    env, fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    time::Instant,
};
const SUITE: &str = include_str!("../../convex_stress/suite.json");
const SCORER: &str = include_str!(
    "../../../.lake/reproduction/gpt/research/complex_geometry_suite/native/src/metrics.rs"
);

fn number(row: &Value, key: &str) -> Result<usize> {
    Ok(row["case"][key]
        .as_u64()
        .ok_or_else(|| format!("missing {key}"))? as usize)
}
fn publish(file: &Path, body: &[u8]) -> Result<()> {
    let temp = file.with_extension("pending");
    let mut handle = fs::File::create(&temp)?;
    handle.write_all(body)?;
    handle.sync_all()?;
    fs::rename(temp, file)?;
    fs::File::open(file.parent().ok_or("missing output directory")?)?.sync_all()?;
    Ok(())
}

fn checkpoint_path(dir: &Path, step: usize, rolling: bool) -> PathBuf {
    let prefix = if rolling { "rolling" } else { "at" };
    dir.join(format!("{prefix}-{step}.safetensors"))
}

fn file_hash(path: &Path) -> Result<String> {
    let mut input = fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let count = input.read(&mut buffer)?;
        if count == 0 {
            return Ok(digest
                .finalize()
                .iter()
                .map(|x| format!("{x:02x}"))
                .collect());
        }
        digest.update(&buffer[..count]);
    }
}

enum Action {
    Run(Option<usize>),
    Replay,
    Depth(Vec<usize>),
}

fn run(action: Action, name: &str, seed: u32, out: &Path) -> Result<Value> {
    let suite: Value = serde_json::from_str(SUITE)?;
    let row = suite["tasks"]
        .as_array()
        .ok_or("empty GPT suite")?
        .iter()
        .find(|r| r["name"] == name)
        .ok_or("unknown canonical GPT task")?;
    let (budget, batch, every) = (
        number(row, "max_updates")?,
        number(row, "batch")?,
        number(row, "observe")?,
    );
    let until = if let Action::Run(Some(stop)) = action {
        stop
    } else {
        budget
    };
    if seed > 2 || until == 0 || until > budget || (until != budget && !until.is_multiple_of(every))
    {
        return Err("expected seed 0-2 and a scheduled observation within the original cap".into());
    }
    let task = data::build(row)?;
    let mut net = engine::Engine::new(
        number(row, "width")?,
        number(row, "heads")?,
        task.vocab,
        seed,
    )?;
    let parameters: usize = net
        .variables
        .trainable_variables()
        .iter()
        .map(|a| a.numel())
        .sum();
    let mut adam = optimizer::Adam::new(&net.variables)?;
    let dir = out.join(name).join(format!("seed{seed}"));
    fs::create_dir_all(&dir)?;
    let lock = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(dir.join("cell.lock"))?;
    lock.lock()?;
    let latest = dir.join("latest.json");
    let bind = hash(&serde_json::to_vec(&json!([
        text::MODEL,
        row,
        seed,
        hash(SCORER.as_bytes())
    ]))?);
    let previous: Option<Value> = if latest.try_exists()? {
        Some(serde_json::from_slice(&fs::read(&latest)?)?)
    } else {
        None
    };
    let mut first = 0usize;
    let mut history = Vec::new();
    let mut elapsed = 0.;
    let primary = &task.splits[&task.selection];
    let previous_rolling = previous
        .as_ref()
        .is_some_and(|v| v["rolling_checkpoint"] == true);
    let rolling = env::var("INKLING_ROLLING").as_deref() == Ok("1");
    if let Some(v) = &previous {
        first = v["step"].as_u64().ok_or("missing checkpoint step")? as usize;
        let checkpoint = checkpoint_path(&dir, first, previous_rolling);
        if v["binding"] != bind || v["checkpoint_sha256"] != file_hash(&checkpoint)? {
            return Err("checkpoint or protocol changed".into());
        }
        net.variables.load(&checkpoint)?;
        elapsed = v["seconds"].as_f64().ok_or("missing training time")?;
        history = v["observations"]
            .as_array()
            .ok_or("missing checkpoint history")?
            .clone();
        if let Action::Depth(budgets) = &action {
            return depth::evaluate(&mut net, &task, v, budgets);
        }
        if matches!(action, Action::Replay) {
            let mut splits = serde_json::Map::new();
            for (key, examples) in &task.splits {
                if key != "train" {
                    let score = metrics::evaluate(&mut net, examples, true)?;
                    splits.insert(
                        key.clone(),
                        json!({"exact":score["exact"],"total":score["examples"]}),
                    );
                }
            }
            let primary_score = &splits[&task.selection];
            if primary_score["exact"] != v["exact"] || primary_score["total"] != v["total"] {
                return Err("independent reload disagreed with the primary observation".into());
            }
            return Ok(
                json!({"model":text::MODEL,"case":name,"seed":seed,"step":first,
                "checkpoint_sha256":v["checkpoint_sha256"],"primary":task.selection,"splits":splits,
                "status":v["status"],"binding":bind}),
            );
        }
        if v["status"] != "in progress" || until == first {
            return Ok(v.clone());
        }
        if until < first {
            return Err("requested observation precedes the latest checkpoint".into());
        }
    } else if !matches!(action, Action::Run(_)) {
        return Err("missing training checkpoint".into());
    }
    let schedule = schedule::Schedule::new(budget)?;
    let train = &task.splits["train"];
    let per_epoch = train.len().div_ceil(batch);
    let mut rng = Random::new(seed + 10_000);
    let mut epoch: Vec<u32> = Vec::new();
    // Reconstitute the original epoch stream, including short final minibatches.
    let mut start_epoch = |epoch: &mut Vec<u32>| {
        epoch.clear();
        epoch.extend(0..train.len() as u32);
        rng.shuffle(epoch);
    };
    for _ in 0..first.div_ceil(per_epoch) {
        start_epoch(&mut epoch);
    }
    let mut packed = metrics::Packed::default();
    let started = Instant::now();
    for step in first..=until {
        if (step == budget || step.is_multiple_of(every)) && (previous.is_none() || step != first) {
            let score = metrics::evaluate(&mut net, primary, true)?;
            let exact = score["exact"]
                .as_u64()
                .ok_or("missing original sequence score")? as usize;
            let pass = exact * 100 >= 99 * primary.len();
            history.push(json!({"step":step,"exact":exact,"total":primary.len()}));
            if pass || step == until {
                let checkpoint = checkpoint_path(&dir, step, rolling);
                let temp = dir.join("writing.safetensors");
                net.variables.save(&temp)?;
                fs::File::open(&temp)?.sync_all()?;
                fs::rename(&temp, &checkpoint)?;
                let mut record = json!({"model":text::MODEL,"case":name,"tier":row["tier"],"seed":seed,
                        "status":if pass{"passed"}else if step==budget{"budget miss"}else{"in progress"},
                        "step":step,"exact":exact,"total":primary.len(),"primary":task.selection,
                        "original_cap":budget,"cadence":every,"parameters":parameters,
                    "stored_layers":4*text::GROUP,"layer_applications":8*text::GROUP,"passes":text::PASSES,"experts":text::EXPERTS,"top_k":text::TOP,
                        "binding":bind,"checkpoint_sha256":file_hash(&checkpoint)?,"observations":history,
                        "seconds":elapsed+started.elapsed().as_secs_f64(),"kernite":"0.6.0","libtorch":"2.9.0 CPU"});
                if rolling {
                    record["rolling_checkpoint"] = json!(true);
                }
                publish(&latest, &serde_json::to_vec_pretty(&record)?)?;
                if rolling && previous_rolling && first != step {
                    fs::remove_file(checkpoint_path(&dir, first, true))?;
                }
                return Ok(record);
            }
        }
        if step.is_multiple_of(per_epoch) {
            start_epoch(&mut epoch);
        }
        let lo = (step % per_epoch) * batch;
        let indices = &epoch[lo..(lo + batch).min(epoch.len())];
        if text::RELEASE {
            let length = indices
                .iter()
                .map(|&i| train[i as usize].tokens.len())
                .max()
                .unwrap();
            let total: usize = indices
                .iter()
                .map(|&i| {
                    train[i as usize]
                        .targets
                        .iter()
                        .filter(|&&y| y >= 0)
                        .count()
                })
                .sum();
            if total == 0 {
                return Err("training minibatch has no targets".into());
            }
            for micro in indices.chunks((256 / length).max(1)) {
                packed.fill(train, micro);
                let targets = packed
                    .labels
                    .iter()
                    .filter(|&&y| (y as usize) < task.vocab)
                    .count();
                if targets > 0 {
                    (net.train_loss(&packed, micro.len())? * (targets as f64 / total as f64))
                        .backward();
                }
            }
        } else {
            packed.fill(train, indices);
            net.train_loss(&packed, indices.len())?.backward();
        }
        adam.step(schedule.rate(step), step)?;
    }
    unreachable!()
}

fn main() -> Result<()> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args == ["model-id"] {
        println!("{}", text::MODEL);
        return Ok(());
    }
    let (action, name, seed, out) = match args.as_slice() {
        [op, name, seed, out] if op == "run" => (Action::Run(None), name, seed, out),
        [op, name, seed, out, cutoff] if op == "run" => (Action::Run(Some(cutoff.parse()?)), name, seed, out),
        [op, name, seed, out] if op == "replay" => (Action::Replay, name, seed, out),
        [op, name, seed, out, counts] if op == "depth" => {
            let budgets = counts.split(',').map(str::parse::<std::num::NonZeroUsize>)
                .collect::<std::result::Result<Vec<_>,_>>()?.into_iter().map(|n| n.get()).collect();
            (Action::Depth(budgets), name, seed, out)
        }
        _ => return Err("usage: inkling-stress run CASE SEED OUT [CUTOFF] | replay CASE SEED OUT | depth CASE SEED OUT PASSES,...".into()),
    };
    let threads = env::var("INKLING_THREADS")
        .unwrap_or_else(|_| "2".into())
        .parse::<i32>()?;
    if threads < 1 {
        return Err("INKLING_THREADS must be positive".into());
    }
    tch::set_num_threads(threads);
    tch::set_num_interop_threads(1);
    let v = run(action, name, seed.parse()?, Path::new(out))?;
    println!("{v}");
    Ok(())
}

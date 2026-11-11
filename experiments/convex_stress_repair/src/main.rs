use convex_stress_minibatch::persistent::Persistent;
use convex_stress_repair::{
    Aligned, MemoryTrainer, Result, SharedModel, data, features::Adapter, hash, input::Observation,
    metrics, random::Random,
};
use serde_json::{Value, json};
use std::{fs, io::Write, path::Path, time::Instant};

const SUITE: &str = include_str!("../../convex_stress/suite.json");
const PROTOCOL: &str = include_str!("../protocol.json");
fn fingerprint() -> String {
    let sources: &[&[u8]] = &[
        include_bytes!("main.rs"),
        include_bytes!("lib.rs"),
        include_bytes!("../Cargo.toml"),
        include_bytes!("../Cargo.lock"),
        include_bytes!("../run_pilot.sh"),
        include_bytes!("../../convex_stress_minibatch/src/lib.rs"),
        include_bytes!("../../convex_stress_minibatch/src/input.rs"),
        include_bytes!("../../convex_stress_minibatch/src/persistent.rs"),
        include_bytes!("../../convex_stress_minibatch/src/metrics.rs"),
        include_bytes!("../../convex_stress/src/features.rs"),
        include_bytes!("../../convex_stress/src/channel.rs"),
        include_bytes!("../../convex_stress/src/vendor/data.rs"),
        include_bytes!("../../convex_stress/src/vendor/arithmetic_data.rs"),
        include_bytes!("../../convex_stress/src/vendor/random.rs"),
        include_bytes!("../../convex_stress/src/vendor/pcg.rs"),
        include_bytes!("../../convex_stress/src/vendor/blake.rs"),
        SUITE.as_bytes(),
        PROTOCOL.as_bytes(),
    ];
    let mut bytes = vec![];
    for source in sources {
        bytes.extend_from_slice(&(source.len() as u64).to_le_bytes());
        bytes.extend_from_slice(source);
    }
    hash(&bytes)
}
fn rows() -> Result<Vec<Value>> {
    Ok(serde_json::from_str::<Value>(SUITE)?["tasks"]
        .as_array()
        .ok_or("tasks missing")?
        .clone())
}
fn selected(name: &str) -> Result<Value> {
    rows()?
        .into_iter()
        .find(|r| r["name"] == name)
        .ok_or_else(|| "unknown task".into())
}
fn features(task: &data::Task) -> Aligned {
    Aligned {
        base: Adapter {
            vocab: task.vocab,
            context: task
                .splits
                .values()
                .flatten()
                .map(|e| {
                    e.tokens
                        .len()
                        .max(e.prompt.len() + e.limit.saturating_sub(1))
                })
                .max()
                .unwrap(),
        },
    }
}
fn encoded(features: Aligned, examples: &[data::Example]) -> (Vec<Observation>, Vec<Vec<usize>>) {
    let mut records = vec![];
    let mut indices = vec![];
    for e in examples {
        let mut ids = vec![];
        for (position, &label) in e.targets.iter().enumerate() {
            if label < 0 {
                continue;
            }
            ids.push(records.len());
            records.push(features.observation(&e.tokens[..=position], label as usize));
        }
        indices.push(ids);
    }
    (records, indices)
}
fn passes(primary: &Value) -> bool {
    !primary.is_null()
        && primary["exact"].as_u64().unwrap() * 100 >= primary["examples"].as_u64().unwrap() * 99
}
fn trace(
    file: &mut fs::File,
    started: Instant,
    update: usize,
    primary: &Value,
    gap: Option<f64>,
) -> Result<()> {
    writeln!(
        file,
        "{}",
        json!({"update":update,"elapsed_seconds":started.elapsed().as_secs_f64(),"primary":primary,"fresh_fw_gap":gap})
    )?;
    file.flush()?;
    println!(
        "update={update} exact={}/{} gap={gap:?}",
        primary["exact"], primary["examples"]
    );
    Ok(())
}
fn run(arm: &str, name: &str, seed: u32, scale: f64, out: &Path) -> Result<()> {
    let protocol: Value = serde_json::from_str(PROTOCOL)?;
    if !["full_batch", "minibatch", "forward_only"].contains(&arm)
        || seed > 2
        || ![1., 10.].contains(&scale)
    {
        return Err("invalid arm, seed or scale".into());
    }
    let row = selected(name)?;
    let task = data::build(&row)?;
    let features = features(&task);
    let (records, indices) = encoded(features, &task.splits["train"]);
    fs::create_dir(out)?;
    let started = Instant::now();
    let batch = row["case"]["batch"].as_u64().unwrap() as usize;
    let per_epoch = indices.len().div_ceil(batch);
    let cadence = (row["case"]["observe"].as_u64().unwrap() as usize).min(per_epoch);
    let mut result = json!({"arm":arm,"task":name,"tier":row["tier"],"seed":seed,"scale":scale,
        "source_fingerprint":fingerprint(),"protocol":protocol,"data_fingerprints":row["data_fingerprints"],
        "batch_size":batch,"selection":task.selection,"train_prefixes":records.len(),"updates_per_epoch":per_epoch,
        "first_primary_99_update":null});
    fs::write(
        out.join("metadata.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    let mut trace_file = fs::File::create(out.join("trace.jsonl"))?;
    let mut rng = Random::new(seed + 10_000);
    let mut order: Vec<_> = (0..indices.len()).collect();
    let model = if arm == "forward_only" {
        rng.shuffle(&mut order);
        let mut trainer = Persistent::new(features.storage());
        for chunk in order.chunks(batch) {
            trainer.update_batch(
                &chunk
                    .iter()
                    .flat_map(|&i| &indices[i])
                    .map(|&i| &records[i])
                    .collect::<Vec<_>>(),
            );
        }
        let mut reference = Persistent::new(features.storage());
        reference.update_batch(&records.iter().collect::<Vec<_>>());
        let actual = trainer.snapshot();
        let expected = reference.snapshot();
        let equal = match (&actual, &expected) {
            (Ok(a), Ok(b)) => serde_json::to_vec(a)? == serde_json::to_vec(b)?,
            (Err(a), Err(b)) => a == b,
            _ => false,
        };
        if !equal {
            return Err("persistent/full-batch forward mismatch".into());
        }
        result["equivalence"] = json!({"equal":equal});
        result["updates"] = trainer.updates.into();
        result["training"] = json!({"observations":trainer.observations});
        match actual {
            Ok(channel) => {
                result["stop_reason"] = "forward_only_complete".into();
                Some(SharedModel {
                    features,
                    scale,
                    channel,
                })
            }
            Err(failed) => {
                result["stop_reason"] = "no_consistent_key".into();
                result["failed_queries"] = json!(failed);
                None
            }
        }
    } else {
        let limit = protocol[if arm == "full_batch" {
            "full_updates"
        } else {
            "minibatch_updates"
        }]
        .as_u64()
        .unwrap() as usize;
        let mut trainer = MemoryTrainer::new(features, scale, records.len());
        let reason = 'training: loop {
            for (i, slot) in order.iter_mut().enumerate() {
                *slot = i;
            }
            rng.shuffle(&mut order);
            let chunk_size = if arm == "full_batch" {
                order.len()
            } else {
                batch
            };
            for chunk in order.chunks(chunk_size) {
                let refs: Vec<_> = chunk
                    .iter()
                    .flat_map(|&i| &indices[i])
                    .map(|&i| (i, &records[i]))
                    .collect();
                trainer.update_batch(&refs);
                if arm == "full_batch"
                    || trainer.updates.is_multiple_of(cadence)
                    || trainer.updates == limit
                {
                    let primary = metrics::evaluate(&trainer, &task.splits[&task.selection])?;
                    let gap = if arm == "full_batch" {
                        Some(trainer.objective(&records).1)
                    } else {
                        None
                    };
                    if gap.is_some_and(|g| g < -1e-8) {
                        return Err("negative fresh FW gap".into());
                    }
                    trace(&mut trace_file, started, trainer.updates, &primary, gap)?;
                    if passes(&primary) {
                        result["first_primary_99_update"] = trainer.updates.into();
                        break 'training "primary_99_percent";
                    }
                    if gap.is_some_and(|g| g <= protocol["full_gap_tolerance"].as_f64().unwrap()) {
                        break 'training "empirical_gap_converged";
                    }
                }
                if trainer.updates == limit {
                    break 'training "diagnostic_budget_exhausted";
                }
            }
        };
        let (ce, gap) = trainer.objective(&records);
        if gap < -1e-8 {
            return Err("negative endpoint FW gap".into());
        }
        result["updates"] = trainer.updates.into();
        result["stop_reason"] = reason.into();
        result["training"] = json!({"mean_ce":ce,"mean_fw_gap":gap,"observations":trainer.observations,"unique_prefixes_seen":trainer.seen()});
        Some(trainer.snapshot())
    };
    if let Some(model) = model {
        let error = model.channel.feasibility_error()?;
        if error > 1e-9 {
            return Err(format!("infeasible model: {error}").into());
        }
        let bytes = serde_json::to_vec(&model)?;
        fs::write(out.join("model.json"), &bytes)?;
        result["model_sha256"] = hash(&bytes).into();
        result["feasibility_error"] = error.into();
        let mut splits = serde_json::Map::new();
        for (name, examples) in &task.splits {
            splits.insert(name.clone(), metrics::evaluate(&model, examples)?);
        }
        result["splits"] = splits.into();
    } else {
        result["splits"] = Value::Null;
        result["model_sha256"] = Value::Null;
    }
    result["status"] = "complete".into();
    result["elapsed_seconds"] = started.elapsed().as_secs_f64().into();
    fs::write(out.join("result.json"), serde_json::to_vec_pretty(&result)?)?;
    Ok(())
}
fn replay(out: &Path) -> Result<()> {
    let result: Value = serde_json::from_slice(&fs::read(out.join("result.json"))?)?;
    if result["source_fingerprint"] != fingerprint() {
        return Err("source fingerprint mismatch".into());
    }
    let row = selected(result["task"].as_str().ok_or("missing task")?)?;
    if result["data_fingerprints"] != row["data_fingerprints"] {
        return Err("data fingerprint mismatch".into());
    }
    let task = data::build(&row)?;
    if result["model_sha256"].is_null() {
        let features = features(&task);
        let (records, _) = encoded(features, &task.splits["train"]);
        let mut trainer = Persistent::new(features.storage());
        trainer.update_batch(&records.iter().collect::<Vec<_>>());
        if json!(
            trainer
                .snapshot()
                .err()
                .ok_or("failure no longer reproduced")?
        ) != result["failed_queries"]
        {
            return Err("failure mismatch".into());
        }
    } else {
        let bytes = fs::read(out.join("model.json"))?;
        if hash(&bytes) != result["model_sha256"] {
            return Err("model hash mismatch".into());
        }
        let model: SharedModel = serde_json::from_slice(&bytes)?;
        if model.channel.feasibility_error()? > 1e-9 {
            return Err("infeasible saved model".into());
        }
        for (name, examples) in &task.splits {
            if metrics::evaluate(&model, examples)? != result["splits"][name] {
                return Err(format!("metric mismatch: {name}").into());
            }
        }
    }
    println!(
        "replay verified {} {} seed {} scale {}",
        result["arm"], result["task"], result["seed"], result["scale"]
    );
    Ok(())
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [command] if command == "fingerprint" => { println!("{}", fingerprint()); Ok(()) },
        [command, arm, task, seed, scale, out] if command == "run" => run(arm, task, seed.parse()?, scale.parse()?, Path::new(out)),
        [command, out] if command == "replay" => replay(Path::new(out)),
        _ => Err("usage: convex-stress-repair fingerprint | run ARM TASK SEED SCALE OUTPUT | replay OUTPUT".into()),
    }
}

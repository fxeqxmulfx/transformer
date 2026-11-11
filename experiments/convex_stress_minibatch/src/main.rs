//! Frozen minibatch comparison on all 35 source stress tasks.
use convex_stress_minibatch::{
    Result,
    channel::Channel,
    data,
    features::Adapter,
    forward, hash, input, metrics,
    persistent::{Persistent, signature},
    random::Random,
    stochastic::Stochastic,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, io::Write, path::Path, time::Instant};
const SUITE: &str = include_str!("../../convex_stress/suite.json");
const PROTOCOL: &str = include_str!("../protocol.json");
fn fingerprint() -> String {
    let sources: &[&[u8]] = &[
        include_bytes!("main.rs"),
        include_bytes!("lib.rs"),
        include_bytes!("input.rs"),
        include_bytes!("stochastic.rs"),
        include_bytes!("persistent.rs"),
        include_bytes!("metrics.rs"),
        include_bytes!("../../convex_stress/src/features.rs"),
        include_bytes!("../../convex_stress/src/channel.rs"),
        include_bytes!("../../convex_stress/src/forward.rs"),
        include_bytes!("../../convex_stress/src/train.rs"),
        include_bytes!("../../convex_stress/src/metrics.rs"),
        include_bytes!("../../convex_stress/src/vendor/data.rs"),
        include_bytes!("../../convex_stress/src/vendor/arithmetic_data.rs"),
        include_bytes!("../../convex_stress/src/vendor/random.rs"),
        include_bytes!("../../convex_stress/src/vendor/pcg.rs"),
        include_bytes!("../../convex_stress/src/vendor/blake.rs"),
        include_bytes!("../Cargo.toml"),
        include_bytes!("../Cargo.lock"),
        include_bytes!("../run_suite.sh"),
        SUITE.as_bytes(),
        PROTOCOL.as_bytes(),
    ];
    let mut sha = Sha256::new();
    for source in sources {
        sha.update((source.len() as u64).to_le_bytes());
        sha.update(source);
    }
    sha.finalize().iter().map(|x| format!("{x:02x}")).collect()
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
        .find(|row| row["name"] == name)
        .ok_or_else(|| "unknown task".into())
}
fn adapter(task: &data::Task) -> Adapter {
    Adapter {
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
    }
}
fn passes(metrics: &Value) -> bool {
    !metrics.is_null()
        && metrics["exact"].as_u64().unwrap() * 100 >= metrics["examples"].as_u64().unwrap() * 99
}
fn verify() -> Result<()> {
    let mut checked = vec![];
    for row in rows()? {
        let task = data::build(&row)?;
        checked.push(json!({"name":row["name"],"tier":row["tier"],"selection":task.selection,
            "adapter":adapter(&task),"splits":task.splits.iter().map(|(k,v)|(k.clone(),v.len())).collect::<BTreeMap<_,_>>()}));
        eprintln!("verified {}", row["name"]);
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({"source_fingerprint":fingerprint(),
        "protocol":serde_json::from_str::<Value>(PROTOCOL)?,"tasks":checked}))?
    );
    Ok(())
}
fn record(
    trace: &mut fs::File,
    started: Instant,
    update: usize,
    primary: &Value,
    failed: &[usize],
) -> Result<()> {
    writeln!(
        trace,
        "{}",
        json!({"update":update,"elapsed_seconds":started.elapsed().as_secs_f64(),
        "primary":primary,"failed_queries":failed})
    )?;
    trace.flush()?;
    println!(
        "update={update} exact={}/{} failures={} seconds={:.2}",
        primary["exact"],
        primary["examples"],
        failed.len(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}
fn save_model(model: &Channel, task: &data::Task, out: &Path, result: &mut Value) -> Result<()> {
    let feasibility = model.feasibility_error()?;
    if feasibility > 1e-9 {
        return Err(format!("channel infeasible: {feasibility}").into());
    }
    let bytes = serde_json::to_vec(model)?;
    fs::write(out.join("channel.json"), &bytes)?;
    let mut splits = serde_json::Map::new();
    for (name, examples) in &task.splits {
        splits.insert(name.clone(), metrics::evaluate(model, examples)?);
    }
    result["splits"] = splits.into();
    result["model_sha256"] = hash(&bytes).into();
    result["feasibility_error"] = feasibility.into();
    Ok(())
}
fn forward_equivalence(
    model: &std::result::Result<Channel, Vec<usize>>,
    task: &data::Task,
    seed: u32,
) -> Result<Value> {
    let (reference, report) = forward::fit(adapter(task), &task.splits["train"], seed);
    match (model, reference) {
        (Ok(model), Some(reference)) => {
            let actual = signature(model).ok_or("non-deterministic forward tensor")?;
            let expected = signature(&reference).ok_or("non-deterministic reference tensor")?;
            if actual != expected {
                return Err("persistent/complete tensor mismatch".into());
            }
            Ok(
                json!({"equal":true,"kind":"entire_joint_tensor","tensor_signature_sha256":hash(&serde_json::to_vec(&actual)?)}),
            )
        }
        (Err(failed), None) if json!(failed) == report["failed_queries"] => {
            Ok(json!({"equal":true,"kind":"explicit_failure","failed_queries":failed}))
        }
        _ => Err("persistent/complete outcome mismatch".into()),
    }
}
fn run(arm: &str, name: &str, seed: u32, out: &Path) -> Result<()> {
    if !["frank_wolfe", "forward_only"].contains(&arm) || seed > 2 {
        return Err("invalid arm or seed".into());
    }
    let row = selected(name)?;
    let task = data::build(&row)?;
    fs::create_dir(out)?;
    let started = Instant::now();
    let mut trace = fs::File::create(out.join("trace.jsonl"))?;
    let batch = row["case"]["batch"].as_u64().ok_or("batch missing")? as usize;
    let max_updates = row["case"]["max_updates"]
        .as_u64()
        .ok_or("budget missing")? as usize;
    let per_epoch = task.splits["train"].len().div_ceil(batch);
    let cadence =
        (row["case"]["observe"].as_u64().ok_or("observe missing")? as usize).min(per_epoch);
    let mut result = json!({"arm":arm,"task":name,"tier":row["tier"],"seed":seed,
        "source_fingerprint":fingerprint(),"suite_sha256":hash(SUITE.as_bytes()),
        "protocol":serde_json::from_str::<Value>(PROTOCOL)?,"batch_size":batch,"max_updates":max_updates,
        "updates_per_epoch":per_epoch,"observe_every":cadence,
        "data_fingerprints":row["data_fingerprints"],"selection":task.selection});
    fs::write(
        out.join("metadata.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    let encoded = input::encode(adapter(&task), &task.splits["train"]);
    let mut order: Vec<_> = (0..encoded.len()).collect();
    let mut rng = Random::new(seed + 10_000);
    let mut first_pass = None;
    if arm == "forward_only" {
        rng.shuffle(&mut order);
        let mut trainer = Persistent::new(adapter(&task));
        for chunk in order.chunks(batch) {
            let samples: Vec<_> = chunk.iter().flat_map(|&i| &encoded[i]).collect();
            trainer.update_batch(&samples);
            if trainer.updates.is_multiple_of(cadence) || trainer.updates == per_epoch {
                match trainer.snapshot() {
                    Ok(model) => {
                        let primary = metrics::evaluate(&model, &task.splits[&task.selection])?;
                        if passes(&primary) {
                            first_pass.get_or_insert(trainer.updates);
                        }
                        record(&mut trace, started, trainer.updates, &primary, &[])?;
                    }
                    Err(failed) => {
                        record(&mut trace, started, trainer.updates, &Value::Null, &failed)?
                    }
                }
            }
        }
        let model = trainer.snapshot();
        result["equivalence"] = forward_equivalence(&model, &task, seed)?;
        result["updates"] = trainer.updates.into();
        result["training"] = json!({"observations":trainer.observations,"candidate_visits":trainer.candidate_visits});
        match model {
            Ok(model) => {
                save_model(&model, &task, out, &mut result)?;
                result["stop_reason"] = "forward_only_complete".into();
            }
            Err(failed) => {
                result["training"]["failed_queries"] = failed.into();
                result["stop_reason"] = "no_consistent_key".into();
                result["splits"] = Value::Null;
                result["model_sha256"] = Value::Null;
            }
        }
    } else {
        let mut trainer = Stochastic::new(adapter(&task));
        let reason = 'training: loop {
            // The frozen source shuffles a fresh ordered range each epoch.
            for (i, index) in order.iter_mut().enumerate() {
                *index = i;
            }
            rng.shuffle(&mut order);
            for chunk in order.chunks(batch) {
                let samples: Vec<_> = chunk.iter().flat_map(|&i| &encoded[i]).collect();
                trainer.update_batch(&samples);
                if trainer.updates.is_multiple_of(cadence) || trainer.updates == max_updates {
                    let primary = metrics::evaluate(&trainer, &task.splits[&task.selection])?;
                    record(&mut trace, started, trainer.updates, &primary, &[])?;
                    if passes(&primary) {
                        first_pass = Some(trainer.updates);
                        break 'training "primary_99_percent";
                    }
                }
                if trainer.updates == max_updates {
                    break 'training "fixed_budget_exhausted";
                }
            }
        };
        let (loss, gap) = trainer.objective(&encoded.iter().flatten().collect::<Vec<_>>());
        result["training"] =
            json!({"observations":trainer.observations,"mean_ce":loss,"mean_fw_gap":gap});
        result["updates"] = trainer.updates.into();
        result["stop_reason"] = reason.into();
        save_model(&trainer.snapshot(), &task, out, &mut result)?;
    }
    result["status"] = "complete".into();
    result["first_primary_99_update"] = json!(first_pass);
    result["elapsed_seconds"] = started.elapsed().as_secs_f64().into();
    fs::write(out.join("result.json"), serde_json::to_vec_pretty(&result)?)?;
    Ok(())
}
fn replay(out: &Path) -> Result<()> {
    let result: Value = serde_json::from_slice(&fs::read(out.join("result.json"))?)?;
    if result["source_fingerprint"] != fingerprint() {
        return Err("source fingerprint mismatch".into());
    }
    let row = selected(result["task"].as_str().ok_or("task missing")?)?;
    let task = data::build(&row)?;
    if result["data_fingerprints"] != row["data_fingerprints"] {
        return Err("data metadata mismatch".into());
    }
    if result["stop_reason"] == "no_consistent_key" {
        let seed = result["seed"].as_u64().unwrap() as u32;
        let encoded = input::encode(adapter(&task), &task.splits["train"]);
        let mut order: Vec<_> = (0..encoded.len()).collect();
        Random::new(seed + 10_000).shuffle(&mut order);
        let mut trainer = Persistent::new(adapter(&task));
        for chunk in order.chunks(result["batch_size"].as_u64().unwrap() as usize) {
            trainer.update_batch(&chunk.iter().flat_map(|&i| &encoded[i]).collect::<Vec<_>>());
        }
        let model = trainer.snapshot();
        if model.is_ok()
            || forward_equivalence(&model, &task, seed)? != result["equivalence"]
            || json!(model.unwrap_err()) != result["training"]["failed_queries"]
        {
            return Err("failure replay mismatch".into());
        }
    } else {
        let bytes = fs::read(out.join("channel.json"))?;
        if hash(&bytes) != result["model_sha256"] {
            return Err("model hash mismatch".into());
        }
        let model: Channel = serde_json::from_slice(&bytes)?;
        if model.feasibility_error()? > 1e-9 {
            return Err("infeasible saved channel".into());
        }
        for (split, examples) in &task.splits {
            if metrics::evaluate(&model, examples)? != result["splits"][split] {
                return Err(format!("saved metric mismatch on {split}").into());
            }
        }
        if result["arm"] == "forward_only"
            && forward_equivalence(&Ok(model), &task, result["seed"].as_u64().unwrap() as u32)?
                != result["equivalence"]
        {
            return Err("saved forward equivalence mismatch".into());
        }
    }
    println!(
        "replay verified {} {} seed {}",
        result["arm"], result["task"], result["seed"]
    );
    Ok(())
}
fn report(root: &Path) -> Result<()> {
    let mut cells = vec![];
    let mut totals = BTreeMap::<String, (usize, usize)>::new();
    for arm in ["frank_wolfe", "forward_only"] {
        for row in rows()? {
            let name = row["name"].as_str().unwrap();
            for seed in 0..3 {
                let path = root.join(arm).join(name).join(format!("seed{seed}"));
                let result: Value = serde_json::from_slice(&fs::read(path.join("result.json"))?)?;
                if result["status"] != "complete"
                    || result["source_fingerprint"] != fingerprint()
                    || result["task"] != name
                    || result["seed"] != seed
                    || result["arm"] != arm
                    || result["data_fingerprints"] != row["data_fingerprints"]
                {
                    return Err("cell metadata mismatch".into());
                }
                if result["stop_reason"] != "no_consistent_key" {
                    if hash(&fs::read(path.join("channel.json"))?) != result["model_sha256"] {
                        return Err("cell model hash mismatch".into());
                    }
                } else if arm != "forward_only"
                    || result["training"]["failed_queries"]
                        .as_array()
                        .is_none_or(|x| x.is_empty())
                {
                    return Err("invalid explicit failure".into());
                }
                let primary =
                    &result["splits"][result["selection"].as_str().ok_or("selection missing")?];
                let pass = passes(primary);
                let total = totals
                    .entry(format!("{arm}/{}", row["tier"].as_str().unwrap()))
                    .or_default();
                total.0 += usize::from(pass);
                total.1 += 1;
                cells.push(json!({"arm":arm,"task":name,"tier":row["tier"],"seed":seed,"pass":pass,"primary":primary,
                    "updates":result["updates"],"first_primary_99_update":result["first_primary_99_update"],
                    "stop_reason":result["stop_reason"],"training":result["training"],"equivalence":result["equivalence"],
                    "feasibility_error":result["feasibility_error"],"elapsed_seconds":result["elapsed_seconds"]}));
            }
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"source_fingerprint":fingerprint(),"totals":totals,"cells":cells})
        )?
    );
    Ok(())
}
fn dispatch() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("verify") if args.len()==2 => verify(),
        Some("run") if args.len()==6 => run(&args[2],&args[3],args[4].parse()?,Path::new(&args[5])),
        Some("replay") if args.len()==3 => replay(Path::new(&args[2])),
        Some("report") if args.len()==3 => report(Path::new(&args[2])),
        _ => Err("usage: convex-stress-minibatch verify | run ARM TASK SEED OUT | replay OUT | report ROOT".into()),
    }
}
fn main() -> std::process::ExitCode {
    match dispatch() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            std::process::ExitCode::FAILURE
        }
    }
}

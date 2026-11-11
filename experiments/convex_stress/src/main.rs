//! Native evaluation of the convex channel family on the frozen GPT stress suite.
#[expect(
    dead_code,
    reason = "Frozen data source also contains arithmetic-only backend entry points"
)]
#[path = "vendor/arithmetic_data.rs"]
mod arithmetic_data;
#[path = "vendor/blake.rs"]
mod blake;
mod channel;
#[path = "vendor/data.rs"]
mod data;
mod features;
mod forward;
mod metrics;
#[path = "vendor/pcg.rs"]
mod pcg;
#[expect(
    dead_code,
    reason = "Preserve frozen RNG including the unused Gaussian entry point"
)]
#[path = "vendor/random.rs"]
mod random;
mod train;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, io::Write, path::Path, time::Instant};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}
const SUITE: &str = include_str!("../suite.json");
const PROTOCOL: &str = include_str!("../protocol.json");
fn fingerprint() -> String {
    let sources: &[&[u8]] = &[
        include_bytes!("main.rs"),
        include_bytes!("features.rs"),
        include_bytes!("channel.rs"),
        include_bytes!("train.rs"),
        include_bytes!("forward.rs"),
        include_bytes!("metrics.rs"),
        include_bytes!("vendor/data.rs"),
        include_bytes!("vendor/arithmetic_data.rs"),
        include_bytes!("vendor/random.rs"),
        include_bytes!("vendor/pcg.rs"),
        include_bytes!("vendor/blake.rs"),
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
fn adapter(task: &data::Task) -> features::Adapter {
    features::Adapter {
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
fn verify() -> Result<()> {
    let mut checked = vec![];
    for row in rows()? {
        let task = data::build(&row)?;
        checked.push(json!({"name":row["name"],"tier":row["tier"],"selection":task.selection,
            "vocab":task.vocab,"adapter":adapter(&task),"splits":task.splits.iter().map(|(k,v)|(k.clone(),v.len())).collect::<std::collections::BTreeMap<_,_>>()}));
        eprintln!("verified {}", row["name"]);
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"source_fingerprint":fingerprint(),"protocol":serde_json::from_str::<Value>(PROTOCOL)?,"tasks":checked})
        )?
    );
    Ok(())
}
fn run(arm: &str, name: &str, seed: u32, out: &Path) -> Result<()> {
    if !["frank_wolfe", "forward_only"].contains(&arm) {
        return Err("arm must be frank_wolfe or forward_only".into());
    }
    if seed > 2 {
        return Err("seed must be 0, 1 or 2".into());
    }
    let row = selected(name)?;
    let task = data::build(&row)?;
    let started = Instant::now();
    fs::create_dir(out)?; // Never silently overwrite an incomplete or completed cell.
    let mut trace = fs::File::create(out.join("trace.jsonl"))?;
    let batch = row["case"]["batch"].as_u64().ok_or("batch missing")? as usize;
    let updates = row["case"]["max_updates"]
        .as_u64()
        .ok_or("budget missing")? as usize;
    let per_epoch = task.splits["train"].len().div_ceil(batch);
    let max_passes = (updates / per_epoch).max(1);
    let header = json!({"arm":arm,"task":name,"tier":row["tier"],"seed":seed,"source_fingerprint":fingerprint(),
        "suite_sha256":hash(SUITE.as_bytes()),"protocol":serde_json::from_str::<Value>(PROTOCOL)?,
        "max_full_passes":max_passes,"original_max_updates":updates,"original_updates_per_epoch":per_epoch,
        "data_fingerprints":row["data_fingerprints"],"selection":task.selection});
    fs::write(
        out.join("metadata.json"),
        serde_json::to_vec_pretty(&header)?,
    )?;
    if arm == "forward_only" {
        let (model, training) = forward::fit(adapter(&task), &task.splits["train"], seed);
        let mut result = header;
        result["status"] = "complete".into();
        result["training"] = training;
        result["full_passes"] = Value::Null;
        if let Some(model) = model {
            save_model(&model, &task, out, &mut result)?;
            result["stop_reason"] = "forward_only_complete".into();
        } else {
            result["stop_reason"] = "no_consistent_key".into();
            result["splits"] = Value::Null;
            result["model_sha256"] = Value::Null;
        }
        result["elapsed_seconds"] = started.elapsed().as_secs_f64().into();
        fs::write(out.join("result.json"), serde_json::to_vec_pretty(&result)?)?;
        println!("{name} seed={seed} forward-only {}", result["stop_reason"]);
        return Ok(());
    }
    let mut trainer = train::Trainer::new(adapter(&task), &task.splits["train"], seed);
    let coverage = trainer.coverage(&task.splits[&task.selection]);
    let mut iteration = 1;
    let reason;
    loop {
        let model = trainer.export();
        let primary = metrics::evaluate(&model, &task.splits[&task.selection])?;
        let pass =
            primary["exact"].as_u64().unwrap() * 100 >= primary["examples"].as_u64().unwrap() * 99;
        let observation = json!({"pass":iteration,"elapsed_seconds":started.elapsed().as_secs_f64(),"primary":primary,
            "fixed_queries":trainer.groups.iter().filter(|g|g.fixed).count(),"queries":trainer.groups.len()});
        writeln!(trace, "{observation}")?;
        trace.flush()?;
        println!(
            "{name} seed={seed} pass={iteration}/{max_passes} exact={}/{} fixed={}/{} seconds={:.1}",
            primary["exact"],
            primary["examples"],
            observation["fixed_queries"],
            trainer.groups.len(),
            started.elapsed().as_secs_f64()
        );
        if pass {
            reason = "primary_99_percent";
            break;
        }
        if trainer.fixed() {
            reason = "training_optimum_certificate";
            break;
        }
        if iteration == max_passes {
            reason = "fixed_budget_exhausted";
            break;
        }
        iteration += 1;
        trainer.step(iteration)?;
    }
    let model = trainer.export();
    let mut result = header;
    save_model(&model, &task, out, &mut result)?;
    result["status"] = "complete".into();
    result["stop_reason"] = reason.into();
    result["full_passes"] = iteration.into();
    result["elapsed_seconds"] = started.elapsed().as_secs_f64().into();
    result["training"] = trainer.report();
    result["input_coverage"] = coverage;
    fs::write(out.join("result.json"), serde_json::to_vec_pretty(&result)?)?;
    Ok(())
}
fn save_model(
    model: &channel::Channel,
    task: &data::Task,
    out: &Path,
    result: &mut Value,
) -> Result<()> {
    let feasibility = model.feasibility_error()?;
    if feasibility > 1e-9 {
        return Err(format!("channel infeasible: {feasibility}").into());
    }
    let model_bytes = serde_json::to_vec(model)?;
    fs::write(out.join("channel.json"), &model_bytes)?;
    let mut results = serde_json::Map::new();
    for (split, examples) in &task.splits {
        results.insert(split.clone(), metrics::evaluate(model, examples)?);
    }
    result["model_sha256"] = hash(&model_bytes).into();
    result["feasibility_error"] = feasibility.into();
    result["splits"] = results.into();
    Ok(())
}
fn replay(out: &Path) -> Result<()> {
    let result: Value = serde_json::from_slice(&fs::read(out.join("result.json"))?)?;
    if result["source_fingerprint"] != fingerprint() {
        return Err("source fingerprint mismatch".into());
    }
    let row = selected(result["task"].as_str().ok_or("task missing")?)?;
    let task = data::build(&row)?;
    if result["stop_reason"] == "no_consistent_key" {
        let (model, training) = forward::fit(
            adapter(&task),
            &task.splits["train"],
            result["seed"].as_u64().unwrap() as u32,
        );
        if model.is_some() || training != result["training"] {
            return Err("forward failure replay mismatch".into());
        }
        println!(
            "replay verified explicit forward failure {} seed {}",
            result["task"], result["seed"]
        );
        return Ok(());
    }
    let bytes = fs::read(out.join("channel.json"))?;
    if result["model_sha256"] != hash(&bytes) {
        return Err("saved channel hash mismatch".into());
    }
    let model: channel::Channel = serde_json::from_slice(&bytes)?;
    if model.feasibility_error()? > 1e-9 {
        return Err("infeasible saved channel".into());
    }
    for (split, examples) in &task.splits {
        let actual = metrics::evaluate(&model, examples)?;
        if actual != result["splits"][split] {
            return Err(format!("saved metrics mismatch on {split}").into());
        }
    }
    println!("replay verified {} seed {}", result["task"], result["seed"]);
    Ok(())
}
fn report(root: &Path) -> Result<()> {
    let mut cells = vec![];
    let mut totals = std::collections::BTreeMap::<String, (usize, usize)>::new();
    for arm in ["frank_wolfe", "forward_only"] {
        for row in rows()? {
            let name = row["name"].as_str().unwrap();
            for seed in 0..3 {
                let path = root.join(arm).join(name).join(format!("seed{seed}"));
                let result: Value = serde_json::from_slice(&fs::read(path.join("result.json"))?)?;
                if result["source_fingerprint"] != fingerprint()
                    || result["seed"] != seed
                    || result["task"] != name
                    || result["arm"] != arm
                    || result["data_fingerprints"] != row["data_fingerprints"]
                {
                    return Err("cell metadata mismatch".into());
                }
                if result["stop_reason"] != "no_consistent_key" {
                    let bytes = fs::read(path.join("channel.json"))?;
                    if result["model_sha256"] != hash(&bytes) {
                        return Err("cell model hash mismatch".into());
                    }
                } else if arm != "forward_only"
                    || result["training"]["failed_queries"]
                        .as_array()
                        .is_none_or(|x| x.is_empty())
                {
                    return Err("invalid explicit failure".into());
                }
                let selection = result["selection"].as_str().ok_or("selection missing")?;
                let metrics = &result["splits"][selection];
                let pass = !metrics.is_null()
                    && metrics["exact"].as_u64().unwrap() * 100
                        >= metrics["examples"].as_u64().unwrap() * 99;
                let total = totals
                    .entry(format!("{arm}/{}", row["tier"].as_str().unwrap()))
                    .or_default();
                total.0 += usize::from(pass);
                total.1 += 1;
                cells.push(json!({"arm":arm,"task":name,"tier":row["tier"],"seed":seed,"pass":pass,"primary":metrics,
                "full_passes":result["full_passes"],"stop_reason":result["stop_reason"],"training":result["training"],
                "input_coverage":result["input_coverage"],"feasibility_error":result["feasibility_error"]}));
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
        Some("verify") if args.len() == 2 => verify(),
        Some("run") if args.len() == 6 => {
            run(&args[2], &args[3], args[4].parse()?, Path::new(&args[5]))
        }
        Some("replay") if args.len() == 3 => replay(Path::new(&args[2])),
        Some("report") if args.len() == 3 => report(Path::new(&args[2])),
        _ => Err(
            "usage: convex-stress verify | run ARM TASK SEED OUT | replay OUT | report ROOT".into(),
        ),
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

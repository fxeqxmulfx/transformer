use convex_stress_spectral::{
    Model, Result, data, hash, linalg, metrics, random::Random, samples, step,
};
use serde_json::{Value, json};
use std::{fs, io::Write, path::Path, time::Instant};

const SUITE: &str = include_str!("../../convex_stress/suite.json");
const PROTOCOL: &str = include_str!("../protocol.json");

fn fingerprint() -> String {
    let sources: &[&[u8]] = &[
        include_bytes!("main.rs"),
        include_bytes!("lib.rs"),
        include_bytes!("linalg.rs"),
        include_bytes!("../Cargo.toml"),
        include_bytes!("../Cargo.lock"),
        include_bytes!("../../convex_stress_minibatch/src/lib.rs"),
        include_bytes!("../../convex_stress_minibatch/src/metrics.rs"),
        include_bytes!("../../convex_stress/src/channel.rs"),
        include_bytes!("../../convex_stress/src/features.rs"),
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

fn selected(name: &str) -> Result<Value> {
    serde_json::from_str::<Value>(SUITE)?["tasks"]
        .as_array()
        .ok_or("no tasks")?
        .iter()
        .find(|r| r["name"] == name)
        .cloned()
        .ok_or_else(|| "unknown task".into())
}
fn radius(label: &str, vocab: usize) -> Result<f64> {
    match label {
        "sqrt" => Ok((vocab as f64).sqrt()),
        "twice_sqrt" => Ok(2. * (vocab as f64).sqrt()),
        "vocab" => Ok(vocab as f64),
        _ => Err("unknown radius".into()),
    }
}
fn metrics_all(model: &Model, task: &data::Task) -> Result<Value> {
    let mut results = serde_json::Map::new();
    for (name, examples) in &task.splits {
        results.insert(name.clone(), metrics::evaluate(model, examples)?);
    }
    Ok(Value::Object(results))
}
fn allowed(protocol: &Value, key: &str, text: &str) -> bool {
    protocol[key].as_array().unwrap().iter().any(|v| v == text)
}
fn update_cap(protocol: &Value, arm: &str) -> usize {
    protocol[match arm {
        "full_batch" => "full_updates",
        "minibatch" => "minibatch_updates",
        _ => "forward_updates",
    }]
    .as_u64()
    .unwrap() as usize
}

fn run(arm: &str, name: &str, r: &str, out: &Path) -> Result<()> {
    let protocol: Value = serde_json::from_str(PROTOCOL)?;
    if !allowed(&protocol, "arms", arm)
        || !allowed(&protocol, "tasks", name)
        || !allowed(&protocol, "radii", r)
    {
        return Err("cell absent from frozen protocol".into());
    }
    let row = selected(name)?;
    let task = data::build(&row)?;
    let mut model = Model::new(task.vocab, radius(r, task.vocab)?);
    let samples = samples(&model, &task.splits["train"])?;
    let batch = row["case"]["batch"].as_u64().unwrap() as usize;
    let per_epoch = samples.len().div_ceil(batch);
    let cadence = (row["case"]["observe"].as_u64().unwrap() as usize).min(per_epoch);
    let cap = protocol["projection_iterations"].as_u64().unwrap() as usize;
    let tol = protocol["projection_tolerance"].as_f64().unwrap();
    let h = protocol["forward_difference_step"].as_f64().unwrap();
    let updates = update_cap(&protocol, arm);
    let seed = protocol["seed"].as_u64().unwrap() as u32;
    fs::create_dir(out)?;
    let started = Instant::now();
    let mut result = json!({"task":name,"arm":arm,"radius_name":r,"radius":model.radius,"seed":seed,
        "source_fingerprint":fingerprint(),"protocol":protocol,"data_fingerprints":row["data_fingerprints"],
        "train_prefixes":samples.len(),"vocab":task.vocab,"batch_size":batch,"updates_per_epoch":per_epoch,
        "selection":task.selection,"first_primary_99_update":null});
    fs::write(
        out.join("metadata.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    let mut trace = fs::File::create(out.join("trace.jsonl"))?;
    let mut order: Vec<_> = (0..samples.len()).collect();
    let mut rng = Random::new(seed + 10_000);
    let (mut caps, mut iterations, mut max_shrink, mut max_change) = (0, 0, 1_f64, 0_f64);
    let mut presented = 0;
    let mut evaluations = 0;
    for update in 0..=updates {
        if update > 0 {
            let within = (update - 1) % per_epoch;
            if arm == "minibatch" && within == 0 {
                rng.shuffle(&mut order);
            }
            let indices = if arm == "minibatch" {
                &order[within * batch..((within + 1) * batch).min(order.len())]
            } else {
                &order
            };
            let eta = if arm == "minibatch" {
                1. / (1. + (update - 1) as f64 / per_epoch as f64).sqrt()
            } else {
                1.
            };
            let p = step(
                &mut model,
                &samples,
                indices,
                (arm == "forward_only").then_some(h),
                eta,
                cap,
                tol,
            )?;
            caps += usize::from(p.capped);
            iterations += p.iterations;
            max_shrink = max_shrink.max(p.radial_shrink);
            max_change = max_change.max(p.change);
            presented += indices.len();
            if arm == "forward_only" {
                evaluations += 4 * indices.len();
            }
        }
        if update % cadence == 0 || update == updates {
            let primary = metrics::evaluate(&model, &task.splits[&task.selection])?;
            let pass = primary["exact"].as_u64().unwrap() * 100
                >= primary["examples"].as_u64().unwrap() * 99;
            if pass && result["first_primary_99_update"].is_null() {
                result["first_primary_99_update"] = update.into();
            }
            let training = metrics::evaluate(&model, &task.splits["train"])?;
            let report = json!({"update":update,"seconds":started.elapsed().as_secs_f64(),"primary":primary,
                "train":training,"projection_caps":caps,"projection_iterations":iterations,"max_radial_shrink":max_shrink});
            writeln!(trace, "{report}")?;
            trace.flush()?;
            println!(
                "update={update} primary={} train={} projection_caps={caps}",
                primary["sequence_accuracy"], training["sequence_accuracy"]
            );
        }
    }
    let (entry_norm, opnorm) = model.validate()?;
    let mut residual_model = model.clone();
    let all: Vec<_> = (0..samples.len()).collect();
    let residual_projection = step(&mut residual_model, &samples, &all, None, 1., cap, tol)?;
    result["projected_gradient_residual"] =
        linalg::frobenius_distance(&model.entries, &residual_model.entries).into();
    result["residual_projection"] = serde_json::to_value(residual_projection)?;
    result["updates"] = updates.into();
    result["examples_presented"] = presented.into();
    result["forward_training_loss_evaluations"] = evaluations.into();
    result["epochs"] = (presented as f64 / samples.len() as f64).into();
    result["projection_caps"] = caps.into();
    result["projection_iterations"] = iterations.into();
    result["max_radial_shrink"] = max_shrink.into();
    result["max_projection_change"] = max_change.into();
    result["max_entry_norm"] = entry_norm.into();
    result["operator_norm"] = opnorm.into();
    result["metrics"] = metrics_all(&model, &task)?;
    result["elapsed_seconds"] = started.elapsed().as_secs_f64().into();
    let bytes = serde_json::to_vec(&model)?;
    result["model_sha256"] = hash(&bytes).into();
    fs::write(out.join("model.json"), bytes)?;
    fs::write(out.join("result.json"), serde_json::to_vec_pretty(&result)?)?;
    println!(
        "complete {}",
        result["metrics"][&task.selection]["sequence_accuracy"]
    );
    Ok(())
}

fn replay(out: &Path) -> Result<()> {
    let report: Value = serde_json::from_slice(&fs::read(out.join("result.json"))?)?;
    let protocol: Value = serde_json::from_str(PROTOCOL)?;
    if report["source_fingerprint"] != fingerprint() || report["protocol"] != protocol {
        return Err("source/protocol mismatch".into());
    }
    let name = report["task"].as_str().ok_or("missing task")?;
    let arm = report["arm"].as_str().ok_or("missing arm")?;
    let r = report["radius_name"].as_str().ok_or("missing radius")?;
    if !allowed(&protocol, "tasks", name)
        || !allowed(&protocol, "arms", arm)
        || !allowed(&protocol, "radii", r)
        || report["updates"] != update_cap(&protocol, arm)
    {
        return Err("invalid reported cell or endpoint".into());
    }
    let row = selected(name)?;
    let task = data::build(&row)?;
    let bytes = fs::read(out.join("model.json"))?;
    if report["model_sha256"] != hash(&bytes)
        || report["data_fingerprints"] != row["data_fingerprints"]
    {
        return Err("model/data fingerprint mismatch".into());
    }
    let model: Model = serde_json::from_slice(&bytes)?;
    if model.vocab != task.vocab || model.radius != radius(r, task.vocab)? {
        return Err("model configuration mismatch".into());
    }
    let (entry, op) = model.validate()?;
    let metrics = metrics_all(&model, &task)?;
    if report["metrics"] != metrics
        || (report["operator_norm"].as_f64().ok_or("no norm")? - op).abs() > 1e-9
        || (report["max_entry_norm"].as_f64().ok_or("no entry norm")? - entry).abs() > 1e-12
    {
        return Err("saved replay differs from reported measurements".into());
    }
    println!(
        "{}",
        json!({"replayed":true,"source_fingerprint":fingerprint(),"task":name,"arm":arm,"radius":r,
        "primary":metrics[&task.selection]["sequence_accuracy"],"operator_norm":op,"max_entry_norm":entry})
    );
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["fingerprint"] => {
            println!("{}", fingerprint());
            Ok(())
        }
        ["run", arm, task, radius, out] => run(arm, task, radius, Path::new(out)),
        ["replay", out] => replay(Path::new(out)),
        _ => Err(
            "usage: convex-stress-spectral fingerprint | run ARM TASK RADIUS OUT | replay OUT"
                .into(),
        ),
    }
}

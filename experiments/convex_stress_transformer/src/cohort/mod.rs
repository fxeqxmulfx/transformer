//! Full original-scope conventional comparator, frozen after backend validation.
use crate::{
    Result, adam, checkpoint, data, engine, hash, layout::Dims, memory, metrics, mt::Mt, preflight,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    cell::RefCell,
    fs,
    io::{Read, Write},
    path::Path,
};
mod replay;
mod run;
mod summary;
pub use replay::replay;
pub use run::run;
pub use summary::summarize;
#[cfg(test)]
mod tests;
const PROTOCOL: &str = include_str!("../../protocol.json");
pub fn protocol() -> Result<Value> {
    Ok(serde_json::from_str(PROTOCOL)?)
}
pub fn fingerprint() -> String {
    let parent = convex_stress_decode::cohort::fingerprint();
    let mut bytes = vec![];
    for source in [
        include_bytes!("../lib.rs").as_slice(),
        include_bytes!("../main.rs"),
        include_bytes!("../model.rs"),
        include_bytes!("../engine.rs"),
        include_bytes!("../memory.rs"),
        include_bytes!("../preflight.rs"),
        include_bytes!("../checkpoint/mod.rs"),
        include_bytes!("../checkpoint/header.rs"),
        include_bytes!("mod.rs"),
        include_bytes!("run.rs"),
        include_bytes!("replay.rs"),
        include_bytes!("summary.rs"),
        include_bytes!("../../Cargo.toml"),
        include_bytes!("../../Cargo.lock"),
        include_bytes!("../../run_cohort.sh"),
        include_bytes!("../../../convex_stress_hull/limited.sh"),
        include_bytes!("../../../grok_division/src/gemm.rs"),
        include_bytes!("../../../grok_division/src/layout.rs"),
        include_bytes!("../../../grok_division/src/adam.rs"),
        include_bytes!("../../../grok_division/src/mt.rs"),
        preflight::SUITE.as_bytes(),
        PROTOCOL.as_bytes(),
        parent.as_bytes(),
    ] {
        bytes.extend_from_slice(&(source.len() as u64).to_le_bytes());
        bytes.extend_from_slice(source);
    }
    hash(&bytes)
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Case {
    pub task: String,
    pub seed: u32,
}
impl Case {
    pub fn id(&self) -> String {
        format!("transformer--{}--seed{}", self.task, self.seed)
    }
}
pub fn cases() -> Result<Vec<Case>> {
    Ok(preflight::rows()?
        .iter()
        .flat_map(|r| {
            [Case {
                task: r["name"].as_str().unwrap().into(),
                seed: 1,
            }]
        })
        .collect())
}
fn case(id: &str) -> Result<Case> {
    cases()?
        .into_iter()
        .find(|c| c.id() == id)
        .ok_or_else(|| "unplanned transformer case".into())
}
fn number(v: &Value, key: &str) -> Result<f64> {
    v[key]
        .as_f64()
        .filter(|x| x.is_finite())
        .ok_or_else(|| format!("missing finite {key}").into())
}
fn integer(v: &Value, key: &str) -> Result<usize> {
    usize::try_from(v[key].as_u64().ok_or("missing integer")?).map_err(Into::into)
}
pub(super) struct Inputs {
    task: data::Task,
    row: Value,
    dims: Dims,
    microbatch: usize,
    batch: usize,
    cap: u64,
    hyper: adam::Hyper,
    buffer_bytes: usize,
}
fn inputs(c: &Case) -> Result<Inputs> {
    let row = preflight::rows()?
        .into_iter()
        .find(|r| r["name"] == c.task)
        .ok_or("missing original task")?;
    let task = data::build(&row)?;
    configure(task, row)
}
/// Reconstruct original data and check training/replay buffers without creating
/// model parameters, drawing training batches or evaluating predictions.
pub fn preflight(task: &str) -> Result<Value> {
    let c = cases()?
        .into_iter()
        .find(|c| c.task == task)
        .ok_or("unknown original transformer task")?;
    Ok(specification(&c, &inputs(&c)?))
}
fn configure(task: data::Task, row: Value) -> Result<Inputs> {
    let dims = preflight::dimensions(&row, &task)?;
    let batch = integer(&row["case"], "batch")?;
    if batch == 0
        || batch > 65536
        || task.splits["train"].is_empty()
        || task.splits["train"].len() > u32::MAX as usize
        || task.splits["train"].len() != integer(&row, "train_examples")?
        || !task.splits.contains_key(&task.selection)
    {
        return Err("invalid original transformer training shape".into());
    }
    for e in task.splits.values().flatten() {
        engine::validate_example(dims, e)?;
    }
    let microbatch = memory::microbatch(dims, batch)?;
    let buffer_bytes =
        memory::check(dims, microbatch)? + 24 * dims.len() + 4 * dims.ctx * dims.d + 64 * 1024;
    if buffer_bytes > memory::BUFFER_LIMIT {
        return Err("transformer replay buffers exceed two GiB".into());
    }
    let cap = integer(&row["case"], "max_updates")? as u64;
    if !(1..=100000).contains(&cap) {
        return Err("invalid original update cap".into());
    }
    let mut hyper = adam::Hyper::source(number(&row["case"], "weight_decay")?);
    hyper.max_lr = number(&row["case"], "learning_rate")?;
    hyper.warmup = integer(&row["case"], "warmup")? as u64;
    if hyper.max_lr <= 0. || hyper.weight_decay < 0. {
        return Err("invalid original AdamW constants".into());
    }
    Ok(Inputs {
        task,
        row,
        dims,
        microbatch,
        batch,
        cap,
        hyper,
        buffer_bytes,
    })
}
fn specification(c: &Case, x: &Inputs) -> Value {
    let d = x.dims;
    let h = x.hyper;
    json!({"case":c,"id":c.id(),"source_fingerprint":fingerprint(),"protocol":protocol().unwrap(),
        "data_fingerprints":x.row["data_fingerprints"],"tier":x.row["tier"],"original_case":x.row["case"],
        "primary_split":x.task.selection,"training_examples":x.task.splits["train"].len(),
        "dimensions":[d.vocab,d.d,d.heads,d.layers,d.dff,d.ctx],"parameters":d.len(),
        "microbatch":x.microbatch,"batch":x.batch,"update_cap":x.cap,"buffer_bytes_upper_bound":x.buffer_bytes,
        "adam":{"max_lr":h.max_lr,"warmup":h.warmup,"beta1":h.beta1,"beta2":h.beta2,"eps":h.eps,"weight_decay":h.weight_decay}})
}
fn read(path: &Path) -> Result<Value> {
    if fs::metadata(path)?.len() > 64 * 1024 * 1024 {
        return Err("oversize transformer JSON".into());
    }
    Ok(serde_json::from_reader(std::io::BufReader::new(
        fs::File::open(path)?,
    ))?)
}
fn write_new(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut w = std::io::BufWriter::new(fs::File::create_new(path)?);
    serde_json::to_writer(&mut w, value)?;
    w.flush()?;
    Ok(())
}
fn file_hash(path: &Path) -> Result<String> {
    if fs::metadata(path)?.len() > 2 * 1024 * 1024 * 1024 {
        return Err("transformer artifact exceeds the streaming read bound".into());
    }
    let mut r = std::io::BufReader::new(fs::File::open(path)?);
    let mut sha = Sha256::new();
    let mut block = [0u8; 65536];
    loop {
        let n = r.read(&mut block)?;
        if n == 0 {
            break;
        }
        sha.update(&block[..n]);
    }
    Ok(sha.finalize().iter().map(|b| format!("{b:02x}")).collect())
}
fn parent_summary(root: &Path) -> Result<Value> {
    if convex_stress_decode::cohort::fingerprint() != protocol()?["queue_parent_fingerprint"] {
        return Err("decoded queue parent changed".into());
    }
    let summary = convex_stress_decode::cohort::summarize(root)?;
    if summary != read(&root.join("summary.json"))?
        || summary["cells"] != 8
        || summary["choices"].as_array().map(Vec::len) != Some(24)
    {
        return Err("complete unchanged decoded cohort required".into());
    }
    Ok(summary)
}
pub fn prepare(root: &Path, parent: &Path) -> Result<()> {
    if root.exists() {
        return Err("transformer cohort destination already exists".into());
    }
    let previous = parent_summary(parent)?;
    let producer = std::env::var("TRANSFORMER_PRODUCER_COMMIT")?;
    if producer.len() != 40 || !producer.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("invalid producer commit".into());
    }
    let mut specifications = vec![];
    for c in cases()? {
        specifications.push(specification(&c, &inputs(&c)?));
    }
    fs::create_dir(root)?;
    write_new(&root.join("parent-summary.json"), &previous)?;
    write_new(
        &root.join("inventory.json"),
        &json!({"source_fingerprint":fingerprint(),"protocol":protocol()?,
        "producer_commit":producer,"cases":cases()?,"specifications":specifications,
        "parent":{"root":fs::canonicalize(parent)?,"summary_sha256":file_hash(&parent.join("summary.json"))?}}),
    )
}
fn inventory(root: &Path) -> Result<Value> {
    let inv = read(&root.join("inventory.json"))?;
    if inv["source_fingerprint"] != fingerprint()
        || inv["protocol"] != protocol()?
        || inv["cases"] != serde_json::to_value(cases()?)?
    {
        return Err("transformer inventory source/protocol mismatch".into());
    }
    Ok(inv)
}
fn check_spec(inv: &Value, c: &Case, x: &Inputs) -> Result<Value> {
    let spec = specification(c, x);
    let rows = inv["specifications"]
        .as_array()
        .ok_or("missing transformer specifications")?;
    if rows.iter().filter(|r| r["id"] == c.id()).count() != 1
        || rows.iter().find(|r| r["id"] == c.id()) != Some(&spec)
    {
        return Err("original transformer specification differs from inventory".into());
    }
    Ok(spec)
}
fn rng(seed: u32) -> Mt {
    Mt::new(2_000_003u32.wrapping_mul(seed.wrapping_add(1)))
}
fn batch<'a>(x: &'a Inputs, rng: &mut Mt) -> Vec<&'a data::Example> {
    (0..x.batch)
        .map(|_| {
            &x.task.splits["train"][rng.interval(x.task.splits["train"].len() as u32 - 1) as usize]
        })
        .collect()
}
fn evaluate(net: &mut engine::Transformer, examples: &[data::Example]) -> Result<Value> {
    metrics::evaluate(&engine::Predictor(RefCell::new(net)), examples)
}
fn metrics_all(net: &mut engine::Transformer, task: &data::Task) -> Result<Value> {
    let mut out = serde_json::Map::new();
    for (name, examples) in &task.splits {
        out.insert(name.clone(), evaluate(net, examples)?);
    }
    Ok(Value::Object(out))
}
fn observe(step: u64, cap: u64) -> Result<bool> {
    Ok(step == 0
        || step == cap
        || step.is_multiple_of(
            protocol()?["observation_every_updates"]
                .as_u64()
                .ok_or("missing observation cadence")?,
        ))
}
fn disk_reserve(path: &Path) -> Result<()> {
    let out = std::process::Command::new("df")
        .arg("--output=avail")
        .arg("-B1")
        .arg(path)
        .output()?;
    let available = std::str::from_utf8(&out.stdout)?
        .lines()
        .last()
        .ok_or("missing disk forecast")?
        .trim()
        .parse::<u64>()?;
    if !out.status.success() || available < protocol()?["disk_reserve_bytes"].as_u64().unwrap() {
        return Err("transformer disk reserve reached; retain partial run".into());
    }
    Ok(())
}

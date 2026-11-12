//! Frozen CM16 inventory: the fold decoder on the unchanged original pair data.
use crate::{Decoder, Mode, Model, Result, Sample, Shape, preflight, validate_observations};
use convex_stress_relabel::cohort::{number, write_new};
use convex_stress_relabel::{budgets, check_layout, data, hash, ideal_loss};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{fs, path::Path};
mod audit;
mod integrate;
mod replay;
mod run;
mod summary;
pub use integrate::integrate;
pub use replay::replay;
pub use run::run;
pub use summary::summarize;
#[cfg(test)]
mod tests;

const PROTOCOL: &str = include_str!("../../protocol.json");
const SUITE: &str = include_str!("../../../convex_stress/suite.json");

pub fn protocol() -> Result<Value> {
    Ok(serde_json::from_str(PROTOCOL)?)
}
/// Own sources, the frozen data and protocol, and the fingerprints of the two
/// crates whose data, metrics, linear algebra and summaries are reused.
pub fn fingerprint() -> String {
    let shared = convex_stress_relabel::cohort::fingerprint();
    let parent = convex_stress_decode::cohort::fingerprint();
    let mut bytes = vec![];
    for source in [
        include_bytes!("../lib.rs").as_slice(),
        include_bytes!("../main.rs"),
        include_bytes!("../transform.rs"),
        include_bytes!("../projection.rs"),
        include_bytes!("../solver.rs"),
        include_bytes!("mod.rs"),
        include_bytes!("run.rs"),
        include_bytes!("replay.rs"),
        include_bytes!("audit.rs"),
        include_bytes!("summary.rs"),
        include_bytes!("integrate.rs"),
        include_bytes!("../../Cargo.toml"),
        include_bytes!("../../Cargo.lock"),
        include_bytes!("../../run_cohort.sh"),
        include_bytes!("../../../convex_stress_hull/limited.sh"),
        SUITE.as_bytes(),
        PROTOCOL.as_bytes(),
        shared.as_bytes(),
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
    pub arm: String,
    pub decoder: String,
    pub budget: String,
}
impl Case {
    pub fn id(&self) -> String {
        format!(
            "{}--{}--{}--{}",
            self.arm, self.task, self.decoder, self.budget
        )
    }
}
fn text(v: &Value, key: &str) -> Result<String> {
    Ok(v[key]
        .as_str()
        .ok_or_else(|| format!("missing text {key}"))?
        .into())
}
fn count(v: &Value, key: &str) -> Result<usize> {
    Ok(v[key]
        .as_u64()
        .ok_or_else(|| format!("missing count {key}"))? as usize)
}
pub fn cases() -> Result<Vec<Case>> {
    let p = protocol()?;
    let mut result = vec![];
    for task in p["tasks"].as_array().ok_or("missing tasks")? {
        for arm in p["arms"].as_array().ok_or("missing arms")? {
            result.push(Case {
                task: task.as_str().ok_or("invalid task")?.into(),
                arm: arm.as_str().ok_or("invalid arm")?.into(),
                decoder: text(&p, "decoder")?,
                budget: text(&p, "budget")?,
            });
        }
    }
    Ok(result)
}
fn case(id: &str) -> Result<Case> {
    cases()?
        .into_iter()
        .find(|c| c.id() == id)
        .ok_or_else(|| "unplanned paired case".into())
}

struct Inputs {
    task: data::Task,
    row: Value,
    shape: Shape,
    decoder: Decoder,
    observations: Vec<Sample>,
    explicit_bytes: u64,
}
/// The three sizes and the decoder follow from the vocabulary alone and are
/// checked against the frozen buffer bound before any numerical buffer exists.
fn assemble(task: data::Task, row: Value) -> Result<Inputs> {
    let shape = Shape {
        side: task.vocab,
        orbits: task.vocab,
        vocab: task.vocab,
    };
    let limit = protocol()?["explicit_memory_limit_bytes"]
        .as_u64()
        .ok_or("missing explicit memory limit")?;
    let explicit_bytes = preflight(shape, limit)?;
    check_layout(&task)?;
    let observations: Vec<_> = task.splits["train"]
        .iter()
        .map(|e| Sample {
            cell: e.tokens[1] as usize + shape.side * e.tokens[0] as usize,
            label: e.targets[1] as usize,
        })
        .collect();
    validate_observations(shape, &observations)?;
    let decoder = Decoder::fold(shape.vocab);
    decoder.validate(shape)?;
    Ok(Inputs {
        task,
        row,
        shape,
        decoder,
        observations,
        explicit_bytes,
    })
}
fn inputs(c: &Case) -> Result<Inputs> {
    if !cases()?.contains(c) {
        return Err("unplanned paired case".into());
    }
    let suite: Value = serde_json::from_str(SUITE)?;
    let row = suite["tasks"]
        .as_array()
        .ok_or("missing suite tasks")?
        .iter()
        .find(|r| r["name"] == c.task)
        .ok_or("missing original task")?
        .clone();
    assemble(data::build(&row)?, row)
}
/// Batches and observation cadence of the original suite row; update caps of the protocol.
struct Schedule {
    count: usize,
    batch: usize,
    per_epoch: usize,
    cadence: usize,
    limit: usize,
}
fn schedule(c: &Case, x: &Inputs) -> Result<Schedule> {
    let p = protocol()?;
    let observations = x.observations.len();
    let batch = count(&x.row["case"], "batch")?;
    if batch == 0 {
        return Err("empty batch".into());
    }
    let per_epoch = observations.div_ceil(batch);
    let cadence = count(&x.row["case"], "observe")?.min(per_epoch);
    let limit = count(
        &p,
        match Mode::parse(&c.arm)? {
            Mode::Full => "full_updates",
            Mode::Mini => "minibatch_updates",
            Mode::Forward => "forward_updates",
        },
    )?;
    if cadence == 0 || limit == 0 {
        return Err("empty observation cadence or update cap".into());
    }
    Ok(Schedule {
        count: observations,
        batch,
        per_epoch,
        cadence,
        limit,
    })
}
fn task_record(name: &str, x: &Inputs) -> Value {
    json!({"task":name,"shape":x.shape,"latent_states":x.shape.latent_states(),
        "training_examples":x.observations.len(),"data_fingerprints":x.row["data_fingerprints"],
        "selection":x.task.selection,"explicit_bytes_upper_bound":x.explicit_bytes})
}
fn check_task(inv: &Value, c: &Case, x: &Inputs) -> Result<()> {
    let tasks = inv["tasks"].as_array().ok_or("missing task inventory")?;
    if tasks.iter().filter(|t| t["task"] == c.task).count() != 1
        || tasks.iter().find(|t| t["task"] == c.task) != Some(&task_record(&c.task, x))
    {
        return Err("original paired task differs from inventory".into());
    }
    Ok(())
}
fn bytes(path: &Path, limit: u64) -> Result<Vec<u8>> {
    if fs::metadata(path)?.len() > limit {
        return Err("oversize saved paired artifact".into());
    }
    Ok(fs::read(path)?)
}
fn read(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&bytes(path, 64 * 1024 * 1024)?)?)
}
/// 32 bytes of JSON text per saved number, plus one MiB.
fn table_limit(shape: Shape, numbers_per_cell: usize) -> u64 {
    32 * (shape.cells() * numbers_per_cell) as u64 + 1024 * 1024
}
fn inventory(root: &Path) -> Result<Value> {
    let inv = read(&root.join("inventory.json"))?;
    if inv["source_fingerprint"] != fingerprint()
        || inv["protocol"] != protocol()?
        || inv["cases"] != serde_json::to_value(cases()?)?
    {
        return Err("paired inventory source/protocol mismatch".into());
    }
    Ok(inv)
}
pub fn prepare(root: &Path) -> Result<()> {
    if root.exists() {
        return Err("paired destination already exists".into());
    }
    let producer = std::env::var("PAIRED_PRODUCER_COMMIT")?;
    if producer.len() != 40 || !producer.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("invalid paired producer commit".into());
    }
    let mut tasks = vec![];
    for name in protocol()?["tasks"].as_array().ok_or("missing tasks")? {
        let c = cases()?
            .into_iter()
            .find(|c| name == c.task.as_str())
            .ok_or("task without a case")?;
        tasks.push(task_record(&c.task, &inputs(&c)?));
    }
    fs::create_dir(root)?;
    write_new(
        &root.join("inventory.json"),
        &json!({"source_fingerprint":fingerprint(),"protocol":protocol()?,"cases":cases()?,
        "producer_commit":producer,"tasks":tasks}),
    )
}
fn metrics_all(model: &Model, task: &data::Task) -> Result<Value> {
    let mut values = serde_json::Map::new();
    for (name, examples) in &task.splits {
        values.insert(name.clone(), crate::metrics::evaluate(model, examples)?);
    }
    Ok(Value::Object(values))
}
/// Everything about a cell that is fixed before its first update.
fn metadata(c: &Case, inv: &Value, x: &Inputs) -> Result<Value> {
    let s = schedule(c, x)?;
    let (radius, nuclear_budget) = budgets(&c.budget, x.shape.side)?;
    let p = protocol()?;
    Ok(
        json!({"case":c,"id":c.id(),"source_fingerprint":fingerprint(),"protocol":p,
        "producer_commit":inv["producer_commit"],"shape":x.shape,
        "latent_states":x.shape.latent_states(),"decoder_orbit_class":x.decoder.orbit_class,
        "radius":radius,"nuclear_budget":nuclear_budget,
        "data_fingerprints":x.row["data_fingerprints"],"selection":x.task.selection,
        "training_examples":s.count,"explicit_bytes_upper_bound":x.explicit_bytes,
        "seed":p["seed"],"batch_size":s.batch,"updates_per_epoch":s.per_epoch,
        "observation_cadence":s.cadence,"update_limit":s.limit}),
    )
}

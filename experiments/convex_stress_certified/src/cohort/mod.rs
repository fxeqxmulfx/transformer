//! Original-data protocol, using CM16's unchanged constructors and metrics.
use crate::{
    solver::{Solver, State},
    *,
};
pub use convex_stress_paired::cohort::Case;
use convex_stress_paired::metrics;
use convex_stress_relabel::cohort::{number, write_new};
use convex_stress_relabel::{data, hash, ideal_loss, random::Random};
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
pub fn fingerprint() -> String {
    let parent = convex_stress_paired::cohort::fingerprint();
    let mut bytes = vec![];
    for source in [
        include_bytes!("../lib.rs").as_slice(),
        include_bytes!("../main.rs"),
        include_bytes!("../training.rs"),
        include_bytes!("../certificate.rs"),
        include_bytes!("../discovery.rs"),
        include_bytes!("../endpoint.rs"),
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
        include_bytes!("../../limited.sh"),
        SUITE.as_bytes(),
        PROTOCOL.as_bytes(),
        parent.as_bytes(),
    ] {
        bytes.extend_from_slice(&(source.len() as u64).to_le_bytes());
        bytes.extend_from_slice(source);
    }
    hash(&bytes)
}
fn count(v: &Value, key: &str) -> Result<usize> {
    usize::try_from(
        v[key]
            .as_u64()
            .ok_or_else(|| format!("missing count {key}"))?,
    )
    .map_err(Into::into)
}
pub fn cases() -> Result<Vec<Case>> {
    let p = protocol()?;
    let mut out = vec![];
    for task in p["tasks"].as_array().ok_or("missing tasks")? {
        for arm in p["arms"].as_array().ok_or("missing arms")? {
            out.push(Case {
                task: task.as_str().ok_or("invalid task")?.into(),
                arm: arm.as_str().ok_or("invalid arm")?.into(),
                decoder: "fold".into(),
                budget: "rank_one".into(),
            });
        }
    }
    Ok(out)
}
fn case(id: &str) -> Result<Case> {
    cases()?
        .into_iter()
        .find(|c| c.id() == id)
        .ok_or_else(|| "unplanned certified case".into())
}
struct Inputs {
    task: data::Task,
    row: Value,
    training: Training,
    observations: Vec<Observation>,
    explicit_bytes: u64,
}
fn assemble(task: data::Task, row: Value) -> Result<Inputs> {
    let shape = Shape {
        side: task.vocab,
        orbits: task.vocab,
        vocab: task.vocab,
    };
    let explicit_bytes = preflight(shape, RUN_MEMORY_BYTES)?;
    convex_stress_relabel::check_layout(&task)?;
    let observations: Vec<_> = task.splits["train"]
        .iter()
        .map(|e| Observation {
            cell: e.tokens[1] as usize + shape.side * e.tokens[0] as usize,
            label: e.targets[1] as usize,
        })
        .collect();
    let mut training = Training::new(shape, Decoder::fold(shape.vocab))?;
    training.observe(&observations)?;
    if observations.is_empty() || training.unique_observations().len() != observations.len() {
        return Err("cohort requires nonempty distinct training observations".into());
    }
    Ok(Inputs {
        task,
        row,
        training,
        observations,
        explicit_bytes,
    })
}
fn inputs(c: &Case) -> Result<Inputs> {
    if !cases()?.contains(c) {
        return Err("unplanned certified case".into());
    }
    let suite: Value = serde_json::from_str(SUITE)?;
    let row = suite["tasks"]
        .as_array()
        .ok_or("missing suite")?
        .iter()
        .find(|r| r["name"] == c.task)
        .ok_or("missing original task")?
        .clone();
    assemble(data::build(&row)?, row)
}
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
        return Err("empty observation cadence or cap".into());
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
    json!({"task":name,"shape":x.training.shape(),"training_examples":x.observations.len(),
        "data_fingerprints":x.row["data_fingerprints"],"selection":x.task.selection,
        "explicit_bytes_upper_bound":x.explicit_bytes})
}
fn check_task(inv: &Value, c: &Case, x: &Inputs) -> Result<()> {
    let tasks = inv["tasks"].as_array().ok_or("missing task inventory")?;
    if tasks.iter().filter(|t| t["task"] == c.task).count() != 1
        || tasks.iter().find(|t| t["task"] == c.task) != Some(&task_record(&c.task, x))
    {
        return Err("original certified task differs from inventory".into());
    }
    Ok(())
}
fn bytes(path: &Path, limit: u64) -> Result<Vec<u8>> {
    if fs::metadata(path)?.len() > limit {
        return Err("oversize certified artifact".into());
    }
    Ok(fs::read(path)?)
}
fn read(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&bytes(path, 64 * 1024 * 1024)?)?)
}
fn table_limit(shape: Shape, width: usize) -> u64 {
    32 * (shape.cells() * width) as u64 + 1024 * 1024
}
fn inventory(root: &Path) -> Result<Value> {
    let inv = read(&root.join("inventory.json"))?;
    if inv["source_fingerprint"] != fingerprint()
        || inv["protocol"] != protocol()?
        || inv["cases"] != serde_json::to_value(cases()?)?
    {
        return Err("certified inventory source/protocol mismatch".into());
    }
    Ok(inv)
}
pub fn prepare(root: &Path) -> Result<()> {
    if root.exists() {
        return Err("certified destination already exists".into());
    }
    let producer = std::env::var("CERTIFIED_PRODUCER_COMMIT")?;
    if producer.len() != 40 || !producer.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("invalid certified producer commit".into());
    }
    if protocol()?["parent_fingerprint"] != convex_stress_paired::cohort::fingerprint() {
        return Err("frozen parent source has changed".into());
    }
    let mut tasks = vec![];
    for c in cases()?.iter().filter(|c| c.arm == "full_batch") {
        tasks.push(task_record(&c.task, &inputs(c)?));
    }
    fs::create_dir(root)?;
    write_new(
        &root.join("inventory.json"),
        &json!({"source_fingerprint":fingerprint(),
        "protocol":protocol()?,"cases":cases()?,"producer_commit":producer,"tasks":tasks}),
    )
}
fn metadata(c: &Case, inv: &Value, x: &Inputs) -> Result<Value> {
    let s = schedule(c, x)?;
    Ok(
        json!({"case":c,"id":c.id(),"source_fingerprint":fingerprint(),"protocol":protocol()?,
        "producer_commit":inv["producer_commit"],"shape":x.training.shape(),
        "latent_states":x.training.shape().latent_states(),"selection":x.task.selection,
        "data_fingerprints":x.row["data_fingerprints"],"explicit_bytes_upper_bound":x.explicit_bytes,
        "training_examples":s.count,"batch_size":s.batch,"updates_per_epoch":s.per_epoch,
        "observation_cadence":s.cadence,"update_limit":s.limit}),
    )
}
fn discovery(x: &Inputs) -> Result<Discovery> {
    let p = protocol()?;
    discover(
        &x.training,
        Limits {
            rectangles: count(&p, "search_rectangles")? as u64,
            candidates: count(&p, "search_candidates")? as u64,
        },
    )
}
fn metrics_all(model: &Model, task: &data::Task) -> Result<Value> {
    let mut values = serde_json::Map::new();
    for (name, examples) in &task.splits {
        values.insert(name.clone(), metrics::evaluate(model, examples)?);
    }
    Ok(Value::Object(values))
}
fn fit_defect(model: &Model, observations: &[Observation]) -> f64 {
    observations
        .iter()
        .flat_map(|s| {
            model.output[s.cell * model.shape.vocab..(s.cell + 1) * model.shape.vocab]
                .iter()
                .enumerate()
                .map(move |(c, &x)| (x - f64::from(c == s.label)).abs())
        })
        .fold(0., f64::max)
}
fn batch(
    solver: &Solver,
    x: &Inputs,
    s: &Schedule,
    mode: Mode,
    update: usize,
    rng: &mut Random,
    order: &mut [usize],
) -> Vec<usize> {
    let within = update % s.per_epoch;
    if mode == Mode::Mini && within == 0 {
        rng.shuffle(order);
    }
    let selected = if mode == Mode::Mini {
        &order[within * s.batch..((within + 1) * s.batch).min(s.count)]
    } else {
        order
    };
    selected
        .iter()
        .map(|&i| {
            solver
                .sample_index(x.observations[i].cell)
                .expect("canonical observation exists")
        })
        .collect()
}

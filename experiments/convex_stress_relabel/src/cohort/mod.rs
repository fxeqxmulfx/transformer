//! Frozen CM12 inventory, provenance and training-only coding preparation.
use crate::{Coding, Discovery, Result, Sample, check_layout, coding, data, hash, memory_bound};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{fs, path::Path};
pub mod audit;
mod replay;
mod run;
pub use replay::{replay, summarize};
pub use run::run;
#[cfg(test)]
mod tests;

pub const PROTOCOL: &str = include_str!("../../protocol.json");
const SUITE: &str = include_str!("../../../convex_stress/suite.json");

pub fn fingerprint() -> String {
    let sources: &[&[u8]] = &[
        include_bytes!("../main.rs"),
        include_bytes!("../lib.rs"),
        include_bytes!("../coding.rs"),
        include_bytes!("mod.rs"),
        include_bytes!("run.rs"),
        include_bytes!("replay.rs"),
        include_bytes!("audit.rs"),
        include_bytes!("../../run_cohort.sh"),
        include_bytes!("../../Cargo.toml"),
        include_bytes!("../../Cargo.lock"),
        include_bytes!("../../../convex_stress_budgets/src/lib.rs"),
        include_bytes!("../../../convex_stress_schatten/src/lib.rs"),
        include_bytes!("../../../convex_stress_schatten/src/solver.rs"),
        include_bytes!("../../../convex_stress_schatten/src/projection.rs"),
        include_bytes!("../../../convex_stress_moments/src/lib.rs"),
        include_bytes!("../../../convex_stress_moments/src/fft.rs"),
        include_bytes!("../../../convex_stress_moments/src/projection.rs"),
        include_bytes!("../../../convex_stress_moments/src/solver.rs"),
        include_bytes!("../../../convex_stress_spectral/src/lib.rs"),
        include_bytes!("../../../convex_stress_spectral/src/linalg.rs"),
        include_bytes!("../../../convex_stress_minibatch/src/lib.rs"),
        include_bytes!("../../../convex_stress_minibatch/src/metrics.rs"),
        include_bytes!("../../../convex_stress/src/channel.rs"),
        include_bytes!("../../../convex_stress/src/features.rs"),
        include_bytes!("../../../convex_stress/src/vendor/data.rs"),
        include_bytes!("../../../convex_stress/src/vendor/arithmetic_data.rs"),
        include_bytes!("../../../convex_stress/src/vendor/random.rs"),
        include_bytes!("../../../convex_stress/src/vendor/pcg.rs"),
        include_bytes!("../../../convex_stress/src/vendor/blake.rs"),
        SUITE.as_bytes(),
        PROTOCOL.as_bytes(),
    ];
    let mut joined = Vec::new();
    for source in sources {
        joined.extend_from_slice(&(source.len() as u64).to_le_bytes());
        joined.extend_from_slice(source);
    }
    hash(&joined)
}
pub fn protocol() -> Result<Value> {
    Ok(serde_json::from_str(PROTOCOL)?)
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Case {
    pub task: String,
    pub arm: String,
    pub coding: String,
    pub budget: String,
}
impl Case {
    pub fn id(&self) -> String {
        format!(
            "{}--{}--{}--{}",
            self.arm, self.task, self.coding, self.budget
        )
    }
    pub fn limit(&self, p: &Value) -> usize {
        p[match self.arm.as_str() {
            "full_batch" => "full_updates",
            "minibatch" => "minibatch_updates",
            _ => "forward_updates",
        }]
        .as_u64()
        .unwrap() as usize
    }
}
pub fn cases() -> Result<Vec<Case>> {
    let p = protocol()?;
    let mut result = vec![];
    for task in p["tasks"].as_array().unwrap() {
        for arm in p["arms"].as_array().unwrap() {
            for coding in p["codings"].as_array().unwrap() {
                for budget in p["budgets"].as_array().unwrap() {
                    result.push(Case {
                        task: task.as_str().unwrap().into(),
                        arm: arm.as_str().unwrap().into(),
                        coding: coding.as_str().unwrap().into(),
                        budget: budget.as_str().unwrap().into(),
                    });
                }
            }
        }
    }
    Ok(result)
}
pub fn case(id: &str) -> Result<Case> {
    cases()?
        .into_iter()
        .find(|c| c.id() == id)
        .ok_or_else(|| "unplanned cell".into())
}
pub struct Inputs {
    pub task: data::Task,
    pub row: Value,
    pub discovery: Discovery,
    pub coding: Option<Coding>,
    pub samples: Vec<Sample>,
    pub explicit_bytes: u64,
}
fn task_record(name: &str, x: &Inputs) -> Value {
    json!({"task":name,"vocab":x.task.vocab,"train_examples":x.task.splits["train"].len(),"discovery":x.discovery,
        "explicit_bytes_upper_bound":x.explicit_bytes,"data_fingerprints":x.row["data_fingerprints"]})
}
fn check_inventory_task(inv: &Value, name: &str, x: &Inputs) -> Result<()> {
    let tasks = inv["tasks"].as_array().ok_or("missing inventory tasks")?;
    if tasks.iter().filter(|r| r["task"] == name).count() != 1
        || tasks.iter().find(|r| r["task"] == name) != Some(&task_record(name, x))
    {
        return Err("recomputed task/coding differs from frozen inventory".into());
    }
    Ok(())
}
pub fn inputs(case: &Case) -> Result<Inputs> {
    if !cases()?.contains(case) {
        return Err("unplanned cell".into());
    }
    let suite: Value = serde_json::from_str(SUITE)?;
    let row = suite["tasks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["name"] == case.task)
        .ok_or("task absent from original suite")?
        .clone();
    let task = data::build(&row)?;
    let explicit_bytes = memory_bound(task.vocab)?;
    if explicit_bytes > protocol()?["explicit_memory_limit_bytes"].as_u64().unwrap() {
        return Err("explicit memory cap exceeded".into());
    }
    check_layout(&task)?;
    let raw: Vec<_> = task.splits["train"]
        .iter()
        .map(|e| Sample {
            cell: e.tokens[1] as usize + task.vocab * e.tokens[0] as usize,
            label: e.targets[1] as usize,
        })
        .collect();
    let discovery = coding::discover(task.vocab, &raw)?;
    let coding = if case.coding == "identity" {
        Some(Coding::identity(task.vocab))
    } else {
        discovery.coding.clone()
    };
    let samples = if let Some(code) = &coding {
        code.samples(&raw)?
    } else {
        vec![]
    };
    Ok(Inputs {
        task,
        row,
        discovery,
        coding,
        samples,
        explicit_bytes,
    })
}
pub fn number(v: &Value, key: &str) -> Result<f64> {
    v[key]
        .as_f64()
        .filter(|x| x.is_finite())
        .ok_or_else(|| format!("missing finite {key}").into())
}
pub fn read(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}
pub fn write_new(path: &Path, value: &impl Serialize) -> Result<()> {
    let file = fs::File::create_new(path)?;
    let mut writer = std::io::BufWriter::new(file);
    serde_json::to_writer(&mut writer, value)?;
    std::io::Write::flush(&mut writer)?;
    Ok(())
}
pub fn metrics_all(model: &crate::Model, task: &data::Task) -> Result<Value> {
    let mut values = serde_json::Map::new();
    for (name, examples) in &task.splits {
        values.insert(name.clone(), crate::metrics::evaluate(model, examples)?);
    }
    Ok(Value::Object(values))
}
pub fn inventory(root: &Path) -> Result<Value> {
    let value = read(&root.join("inventory.json"))?;
    if value["source_fingerprint"] != fingerprint()
        || value["protocol"] != protocol()?
        || value["cases"] != serde_json::to_value(cases()?)?
    {
        return Err("inventory source/protocol/case mismatch".into());
    }
    Ok(value)
}
pub fn prepare(root: &Path) -> Result<()> {
    if root.exists() {
        return Err("cohort destination already exists".into());
    }
    let p = protocol()?;
    let mut tasks = vec![];
    for name in p["tasks"].as_array().unwrap() {
        let c = cases()?
            .into_iter()
            .find(|c| c.task == name.as_str().unwrap())
            .unwrap();
        let x = inputs(&c)?;
        tasks.push(task_record(&c.task, &x));
    }
    let producer = std::env::var("RELABEL_PRODUCER_COMMIT")?;
    if producer.len() != 40 || !producer.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("invalid producer commit".into());
    }
    fs::create_dir(root)?;
    write_new(
        &root.join("inventory.json"),
        &json!({"source_fingerprint":fingerprint(),"protocol":p,"producer_commit":producer,"cases":cases()?,"tasks":tasks}),
    )
}

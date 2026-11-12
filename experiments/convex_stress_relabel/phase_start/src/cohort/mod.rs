//! Prospective CM14 evaluation with unchanged cold and original-data controls.
pub use cold::cohort::Case;
use cold::cohort::{Inputs, number, read, write_new};
use cold::{Result, Sample, hash};
use convex_stress_relabel as cold;
use serde_json::{Value, json};
use std::{fs, path::Path};
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
    let old = cold::cohort::fingerprint();
    let graph = convex_relabel_graph_audit::fingerprint();
    let mut bytes = vec![];
    for source in [
        include_bytes!("../lib.rs").as_slice(),
        include_bytes!("../main.rs"),
        include_bytes!("mod.rs"),
        include_bytes!("run.rs"),
        include_bytes!("replay.rs"),
        include_bytes!("summary.rs"),
        include_bytes!("../../Cargo.toml"),
        include_bytes!("../../Cargo.lock"),
        include_bytes!("../../run_cohort.sh"),
        PROTOCOL.as_bytes(),
        old.as_bytes(),
        graph.as_bytes(),
    ] {
        bytes.extend_from_slice(&(source.len() as u64).to_le_bytes());
        bytes.extend_from_slice(source);
    }
    hash(&bytes)
}
pub fn cases() -> Result<Vec<Case>> {
    let p = protocol()?;
    let mut out = vec![];
    for task in p["tasks"].as_array().unwrap() {
        for coding in p["codings"].as_array().unwrap() {
            out.push(Case {
                task: task.as_str().unwrap().into(),
                coding: coding.as_str().unwrap().into(),
                arm: "phase_construct".into(),
                budget: "rank_one".into(),
            });
        }
    }
    Ok(out)
}
fn case(id: &str) -> Result<Case> {
    cases()?
        .into_iter()
        .find(|x| x.id() == id)
        .ok_or_else(|| "unplanned phase case".into())
}
fn inputs(c: &Case) -> Result<Inputs> {
    let mut control = c.clone();
    control.arm = "full_batch".into();
    cold::cohort::inputs(&control)
}
fn raw(x: &Inputs) -> Vec<Sample> {
    x.task.splits["train"]
        .iter()
        .map(|e| Sample {
            cell: e.tokens[1] as usize + x.task.vocab * e.tokens[0] as usize,
            label: e.targets[1] as usize,
        })
        .collect()
}
fn task_record(name: &str, x: &Inputs) -> Value {
    json!({"task":name,"vocab":x.task.vocab,"training_examples":x.task.splits["train"].len(),
        "data_fingerprints":x.row["data_fingerprints"],"discovery":x.discovery,"explicit_bytes_upper_bound":x.explicit_bytes})
}
fn check_task(inv: &Value, c: &Case, x: &Inputs) -> Result<()> {
    let tasks = inv["tasks"].as_array().ok_or("missing inventory tasks")?;
    if tasks.iter().filter(|r| r["task"] == c.task).count() != 1
        || tasks.iter().find(|r| r["task"] == c.task) != Some(&task_record(&c.task, x))
    {
        return Err("phase data/discovery differs from frozen inventory".into());
    }
    Ok(())
}
fn inventory(root: &Path) -> Result<Value> {
    let inv = read(&root.join("inventory.json"))?;
    if inv["source_fingerprint"] != fingerprint()
        || inv["protocol"] != protocol()?
        || inv["cases"] != serde_json::to_value(cases()?)?
    {
        return Err("phase inventory source/protocol mismatch".into());
    }
    Ok(inv)
}
fn parent_summaries(cold_root: &Path, graph_root: &Path) -> Result<(Value, Value)> {
    let p = protocol()?;
    if cold::cohort::fingerprint() != p["cold_fingerprint"]
        || convex_relabel_graph_audit::fingerprint() != p["graph_auditor_fingerprint"]
    {
        return Err("frozen dependency fingerprint changed".into());
    }
    let cold_summary = cold::cohort::summarize(cold_root)?;
    let graph = read(&graph_root.join("summary.json"))?;
    if cold_summary != read(&cold_root.join("summary.json"))?
        || cold_summary["cells"] != 96
        || cold_summary["choices"].as_array().map(Vec::len) != Some(24)
        || graph["cells"] != 96
        || graph["auditor_fingerprint"] != p["graph_auditor_fingerprint"]
        || graph["producer_fingerprint"] != p["cold_fingerprint"]
        || graph["producer_summary_sha256"] != hash(&fs::read(cold_root.join("summary.json"))?)
        || graph["complete"] != cold_summary["complete"]
        || graph["proposal_ineligible"] != cold_summary["proposal_ineligible"]
    {
        return Err("complete unchanged cold and graph summaries required".into());
    }
    Ok((cold_summary, graph))
}
pub fn prepare(root: &Path, cold_root: &Path, graph_root: &Path) -> Result<()> {
    if root.exists() {
        return Err("phase destination already exists".into());
    }
    let (cold_summary, graph_summary) = parent_summaries(cold_root, graph_root)?;
    let producer = std::env::var("PHASE_PRODUCER_COMMIT")?;
    if producer.len() != 40 || !producer.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err("invalid producer commit".into());
    }
    let mut tasks = vec![];
    for c in cases()?.into_iter().filter(|c| c.coding == "cycle") {
        tasks.push(task_record(&c.task, &inputs(&c)?));
    }
    let parents = json!({"cold_root":fs::canonicalize(cold_root)?,"graph_root":fs::canonicalize(graph_root)?,
        "cold_summary_sha256":hash(&fs::read(cold_root.join("summary.json"))?),
        "graph_summary_sha256":hash(&fs::read(graph_root.join("summary.json"))?)});
    fs::create_dir(root)?;
    write_new(&root.join("cold-summary.json"), &cold_summary)?;
    write_new(&root.join("graph-summary.json"), &graph_summary)?;
    write_new(
        &root.join("inventory.json"),
        &json!({"source_fingerprint":fingerprint(),"protocol":protocol()?,
        "producer_commit":producer,"parents":parents,"cases":cases()?,"tasks":tasks}),
    )
}
fn construct(x: &Inputs) -> Result<(Option<cold::Model>, Value)> {
    if let Some(code) = &x.coding {
        let p = crate::propose(x.task.vocab, &raw(x), code)?;
        let details = json!({"structure":p.structure,"rejection":p.rejection,
            "phase_examples_presented":p.examples_presented,"fit_verification_examples":if p.model.is_some(){p.examples_presented}else{0}});
        Ok((p.model, details))
    } else {
        Ok((
            None,
            json!({"structure":null,"rejection":"coding_unavailable","phase_examples_presented":0,"fit_verification_examples":0}),
        ))
    }
}

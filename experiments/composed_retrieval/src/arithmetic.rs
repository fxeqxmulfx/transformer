//! Original source rows and refusal accounting for the 54-case extension.
//! Only the resulting aligned examples and object dimensions enter training.
use crate::{
    Result,
    data::{ARITHMETIC_PROTOCOL, CaseSpec, DENSE_PROTOCOL, PROTOCOL, Protocol, TRAINING_PROTOCOL},
};
use arithmetic_data::{dataset, inventory, oracle};
use baseline::{
    data::{Example, Task},
    hash,
};
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::Path};

fn source_cases() -> Result<Vec<Value>> {
    Ok(inventory::cases().map_err(|e| e.to_string())?)
}
fn name(row: &Value) -> Result<&str> {
    Ok(row["case"]["name"].as_str().ok_or("original case name")?)
}
fn count(row: &Value, field: &str) -> Result<usize> {
    Ok(usize::try_from(
        row[field].as_u64().ok_or("original count")?,
    )?)
}
pub fn configure(p: &mut Protocol) -> Result<()> {
    let family: Value = serde_json::from_str(ARITHMETIC_PROTOCOL)?;
    if family["parent_protocol_sha256"] != hash(DENSE_PROTOCOL.as_bytes())
        || family["data_protocol_sha256"] != hash(arithmetic_data::PROTOCOL.as_bytes())
    {
        return Err("changed training or dataset parent".into());
    }
    let mut cases = BTreeMap::new();
    let mut unavailable = vec![];
    for row in source_cases()? {
        let id = name(&row)?;
        let case = &row["case"];
        let history = &row["history"];
        if case["data_seed"] != p.data_seed
            || case["metric"] != "sequence_accuracy"
            || case["target"] != 0.99
            || history["case"] != id
        {
            return Err("arithmetic source identity or criterion changed".into());
        }
        let op = case["data"]["operation"].as_str().ok_or("operation")?;
        if oracle::UNAVAILABLE.contains(&op) {
            unavailable.push(id.to_owned());
            continue;
        }
        cases.insert(
            id.to_owned(),
            CaseSpec {
                vocabulary: count(&case["data"], "prime")?,
                context: 2,
                batch: count(case, "batch")?,
                train: count(&history["train"], "examples")?,
                selection: "validation".into(),
                metric: "sequence_accuracy".into(),
                target: 0.99,
                metadata_sha256: None,
                source_fingerprints: Some(history["data_fingerprints"].clone()),
            },
        );
    }
    unavailable.sort();
    if cases.len() != 48 || family["unavailable"] != json!(unavailable) {
        return Err("arithmetic cohort membership changed".into());
    }
    let protocol = if cfg!(feature = "logodds-depth") {
        let joined: Value = serde_json::from_str(PROTOCOL)?;
        if joined["training_protocol_sha256"] != hash(TRAINING_PROTOCOL.as_bytes())
            || joined["arithmetic_protocol_sha256"] != hash(ARITHMETIC_PROTOCOL.as_bytes())
        {
            return Err("training or arithmetic source protocol changed".into());
        }
        joined
    } else {
        family
    };
    p.name = protocol["name"].as_str().ok_or("extension name")?.into();
    p.cases = cases;
    Ok(())
}
fn examples(rows: Vec<[u32; 3]>) -> Vec<Example> {
    rows.into_iter()
        .map(|[a, b, y]| Example {
            task: "arithmetic".into(),
            tokens: vec![a.into(), b.into()],
            targets: vec![-100, y.into()],
            changes: vec![false, false],
            prompt: vec![],
            answer: vec![],
            limit: 0,
            field: None,
            row_index: None,
        })
        .collect()
}
pub fn original(id: &str) -> Result<Task> {
    let row = source_cases()?
        .into_iter()
        .find(|r| r["case"]["name"] == id)
        .ok_or("arithmetic source case")?;
    let data = dataset::build(&row)
        .map_err(|e| e.to_string())?
        .ok_or("original fixed helper unavailable")?;
    Ok(Task {
        vocab: data.training.vocabulary,
        selection: "validation".into(),
        splits: BTreeMap::from([
            ("train".into(), examples(data.training.rows)),
            ("validation".into(), examples(data.validation)),
        ]),
    })
}

/// Validate all source inputs and splits before fitting, including every
/// unavailable member. Forecast the exact number of raw logit bytes.
pub fn preflight(p: &Protocol, inputs: &Path) -> Result<Value> {
    inventory::check_inputs(inputs).map_err(|e| e.to_string())?;
    let mut cases = BTreeMap::new();
    let mut one_pass = 0_u64;
    for row in source_cases()? {
        let id = name(&row)?;
        let data = dataset::build(&row).map_err(|e| e.to_string())?;
        if p.cases.contains_key(id) != data.is_some() {
            return Err("source support differs from frozen cases".into());
        }
        let split = data.map(|d| {
            let rows = d.training.rows.len()+d.validation.len();
            one_pass += 8 * d.training.vocabulary as u64 * rows as u64;
            json!({"vocabulary":d.training.vocabulary,"train":d.training.rows.len(),"validation":d.validation.len()})
        });
        cases.insert(
            id.to_owned(),
            json!({"dimensions":split,"split_fingerprints":row["history"]["data_fingerprints"]}),
        );
    }
    let fits = p.cases.len() * p.seeds.len() * p.modes.len() * 2;
    let raw_bytes = one_pass * 2 * 2 * p.seeds.len() as u64 * p.modes.len() as u64;
    Ok(
        json!({"protocol_sha256":hash(PROTOCOL.as_bytes()),"cases":cases,"fits":fits,"raw_bytes":raw_bytes}),
    )
}

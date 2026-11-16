//! Complete six-case inventory. A hash mismatch remains an explicit refusal.
//! No model is constructed and no accuracy can be credited by these commands.
use crate::{
    Result, cases,
    dataset::{self, Dataset},
    storage::{self, Artifact, Guard},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{fs, path::Path};
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Entry {
    pub source: Value,
    pub artifact: Option<Artifact>,
    pub summary: Option<Value>,
    pub refusal: Option<String>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Inventory {
    pub schema: u32,
    pub producer: String,
    pub model_updates: u64,
    pub model_measurements: u64,
    pub complete: bool,
    pub cases: Vec<Entry>,
}
pub fn prepare(workspace: &Path, output: &Path, helper: &Path, producer: &str) -> Result<bool> {
    if producer.len() != 40 || !producer.bytes().all(|x| x.is_ascii_hexdigit()) {
        return Err("expected frozen producer commit".into());
    }
    let parent = output.parent().ok_or("output parent")?.canonicalize()?;
    let guard = Guard {
        workspace: workspace.canonicalize()?,
        storage: parent,
    };
    guard.check()?;
    // Refuse overwrite, including partial evidence from an earlier failure.
    fs::create_dir(output)?;
    let mut inventory = Inventory {
        schema: 1,
        producer: producer.into(),
        model_updates: 0,
        model_measurements: 0,
        complete: true,
        cases: vec![],
    };
    for source in cases()? {
        guard.check()?;
        let name = source["case"]["name"].as_str().ok_or("case name")?;
        let entry = match dataset::build(&source, helper) {
            Ok(data) => {
                let artifact = storage::json(&output.join(format!("{name}.json")), &data, &guard)?;
                Entry {
                    source,
                    artifact: Some(artifact),
                    summary: Some(data.summary()?),
                    refusal: None,
                }
            }
            Err(error) => {
                inventory.complete = false;
                Entry {
                    source,
                    artifact: None,
                    summary: None,
                    refusal: Some(error.to_string()),
                }
            }
        };
        inventory.cases.push(entry);
    }
    storage::json(&output.join("inventory.json"), &inventory, &guard)?;
    storage::json(
        &output.join("engine.json"),
        &json!({
            "native_binary":storage::artifact(&storage::read(helper)?),
            "native_path":helper.canonicalize()?,
            "model_updates":0,"model_measurements":0,
        }),
        &guard,
    )?;
    Ok(inventory.complete)
}
pub fn read(root: &Path) -> Result<Inventory> {
    let result: Inventory = serde_json::from_slice(&storage::read(&root.join("inventory.json"))?)?;
    let sources = cases()?;
    if result.schema != 1
        || result.model_updates != 0
        || result.model_measurements != 0
        || result.cases.len() != sources.len()
        || result.complete != result.cases.iter().all(|entry| entry.refusal.is_none())
    {
        return Err("incomplete or inconsistent six-case inventory".into());
    }
    for (entry, source) in result.cases.iter().zip(sources) {
        if entry.source != source {
            return Err("source history or protocol changed".into());
        }
        let name = source["case"]["name"].as_str().ok_or("source name")?;
        let path = root.join(format!("{name}.json"));
        match (&entry.artifact, &entry.summary, &entry.refusal) {
            (Some(expected), Some(summary), None) => {
                let bytes = storage::read(&path)?;
                if storage::artifact(&bytes) != *expected {
                    return Err("dataset artifact changed".into());
                }
                let dataset: Dataset = serde_json::from_slice(&bytes)?;
                dataset.validate(&source)?;
                if dataset.summary()? != *summary {
                    return Err("full dataset summary changed".into());
                }
            }
            (None, None, Some(reason)) if !reason.is_empty() && !path.exists() => {}
            _ => return Err("refusal and prepared dataset status conflict".into()),
        }
    }
    Ok(result)
}
pub fn verify(workspace: &Path, original: &Path, repeat: &Path, output: &Path) -> Result<()> {
    let guard = Guard {
        workspace: workspace.canonicalize()?,
        storage: output.parent().ok_or("output parent")?.canonicalize()?,
    };
    guard.check()?;
    let a = read(original)?;
    let b = read(repeat)?;
    if a != b {
        return Err("independent complete inventories differ".into());
    }
    if storage::read(&original.join("inventory.json"))?
        != storage::read(&repeat.join("inventory.json"))?
    {
        return Err("independent inventory bytes differ".into());
    }
    storage::json(
        output,
        &json!({
            "inventory_sha256":crate::hash(&storage::read(&original.join("inventory.json"))?),
            "cases":a.cases.len(),"accepted":a.cases.iter().filter(|e|e.refusal.is_none()).count(),
            "all_datasets_verified":a.complete,"identical_inventory":true,
            "model_updates":0,"model_measurements":0,
        }),
        &guard,
    )?;
    Ok(())
}

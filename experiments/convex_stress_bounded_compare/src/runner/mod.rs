//! Whole original cohort orchestration. Only this layer sees case identities
//! and historical scores. The numerical model receives Training or a pair.
use crate::*;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Component, Path},
};
pub mod inputs;
pub mod scoring;
pub mod stage;
pub mod summary;
pub const PROTOCOL: &str = include_str!("../../protocol.json");
pub const JSON_LIMIT: u64 = 64 * 1024 * 1024;

pub fn protocol() -> Result<Value> {
    Ok(serde_json::from_str(PROTOCOL)?)
}
pub fn read<T: DeserializeOwned>(path: &Path) -> Result<T> {
    if fs::metadata(path)?.len() > JSON_LIMIT {
        return Err("runner JSON exceeds64MiB".into());
    }
    Ok(serde_json::from_reader(std::io::BufReader::new(
        fs::File::open(path)?,
    ))?)
}
pub fn relative(name: &str) -> Result<()> {
    if name.is_empty()
        || Path::new(name)
            .components()
            .any(|x| !matches!(x, Component::Normal(_)))
    {
        return Err("unsafe artifact-relative path".into());
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Case {
    pub name: String,
    pub existing35: bool,
    pub constant_control: bool,
    pub vocabulary: usize,
    pub training_records: usize,
    pub validation_records: usize,
    pub support: usize,
    pub shape: Option<joint::Shape>,
    pub source_refusal: Option<String>,
    pub forecast: Option<resources::CaseBudget>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Inventory {
    pub protocol: Value,
    pub producer_commit: String,
    pub cases: Vec<Case>,
    pub inputs: BTreeMap<String, disk::Artifact>,
    pub single_run_budget: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Local {
    pub copies_reserved: u64,
    pub binary_sha256: String,
}
pub fn guard(
    root: &Path,
    workspace: &Path,
    inventory: &Inventory,
    copies: u64,
) -> Result<resources::Guard> {
    if ![1, 2].contains(&copies) {
        return Err("invalid original/repeat disk reservation".into());
    }
    let p = protocol()?;
    Ok(resources::Guard {
        workspace: workspace.to_path_buf(),
        storage: root.parent().ok_or("run parent")?.to_path_buf(),
        run: root.to_path_buf(),
        output_budget: inventory
            .single_run_budget
            .checked_mul(copies)
            .ok_or("run budget overflow")?,
        workspace_reserve: p["workspace_disk_reserve_bytes"]
            .as_u64()
            .ok_or("workspace reserve")?,
        storage_reserve: p["storage_disk_reserve_bytes"]
            .as_u64()
            .ok_or("storage reserve")?,
    })
}
pub fn load(root: &Path, workspace: &Path) -> Result<(Inventory, resources::Guard)> {
    let inv: Inventory = read(&root.join("inventory.json"))?;
    let local: Local = read(&root.join("run-local.json"))?;
    if inv.protocol != protocol()?
        || inv.cases.len() != 54
        || inv
            .cases
            .iter()
            .map(|x| &x.name)
            .collect::<BTreeSet<_>>()
            .len()
            != 54
    {
        return Err("original cohort inventory or protocol changed".into());
    }
    if disk::hash(&std::env::current_exe()?)?.sha256 != local.binary_sha256 {
        return Err("run executable changed".into());
    }
    for (name, expected) in inv.protocol["references"]
        .as_object()
        .ok_or("reference hashes")?
    {
        let file = format!("inputs/{name}");
        let saved = inv.inputs.get(&file).ok_or("missing pinned reference")?;
        if saved.sha256 != expected.as_str().ok_or("reference digest")?
            || disk::hash(&root.join(file))? != *saved
        {
            return Err("pinned reference hash changed".into());
        }
    }
    for (path, expected) in &inv.inputs {
        relative(path)?;
        if fs::metadata(root.join(path))?.len() != expected.bytes {
            return Err("input length changed".into());
        }
    }
    let admission = guard(root, workspace, &inv, local.copies_reserved)?;
    admission.check()?;
    inputs::validate(root, &inv)?;
    Ok((inv, admission))
}
pub fn checked<T: DeserializeOwned>(root: &Path, inv: &Inventory, path: &str) -> Result<T> {
    relative(path)?;
    if disk::hash(&root.join(path))? != *inv.inputs.get(path).ok_or("unbound input artifact")? {
        return Err("input hash changed".into());
    }
    read(&root.join(path))
}
pub fn case<'a>(inv: &'a Inventory, name: &str) -> Result<&'a Case> {
    inv.cases
        .iter()
        .find(|x| x.name == name)
        .ok_or_else(|| "case absent from complete original cohort".into())
}
pub fn selection(root: &Path, inv: &Inventory, name: &str) -> Result<pairs::selection::Selection> {
    let value: Value = checked(root, inv, &format!("inputs/probe/{name}.json"))?;
    if value["heldout_pools_read"] != 0
        || value["training_updates"] != 0
        || value["accuracy_measurements"] != 0
    {
        return Err("selection input is not the frozen training-only probe".into());
    }
    Ok(serde_json::from_value(value["selection"].clone())?)
}
pub fn training(root: &Path, inv: &Inventory, name: &str) -> Result<pairs::Training> {
    checked(root, inv, &format!("inputs/train/{name}.json"))
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Dataset {
    pub training: pairs::Training,
    pub validation: Vec<[u32; 3]>,
}
impl Dataset {
    pub fn validate(&self, input: &pairs::Training, case: &Case) -> Result<()> {
        input.validate()?;
        if self.training != *input
            || input.vocabulary != case.vocabulary
            || input.rows.len() != case.training_records
            || self.validation.len() != case.validation_records
        {
            return Err("original split/counts changed".into());
        }
        let mut pairs = BTreeSet::new();
        if input.rows.iter().chain(&self.validation).any(|r| {
            r.iter().any(|&v| v as usize >= case.vocabulary) || !pairs.insert([r[0], r[1]])
        }) {
            return Err("original pair split is not disjoint or in range".into());
        }
        Ok(())
    }
}

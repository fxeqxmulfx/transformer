//! A complete training barrier, validated again before every score or replay.
//! A source refusal occupies its original slot; absent endpoints do not.
use crate::{resources::Guard, schema::Seal, *};
use std::{collections::BTreeMap, path::Path};
pub(crate) fn collect(root: &Path, inputs: &inputs::Inputs) -> Result<Seal> {
    let mut artifacts = BTreeMap::new();
    for path in ["run.json", "data/inventory.json"] {
        artifacts.insert(path.into(), io::artifact(&root.join(path))?);
    }
    let mut entries = vec![];
    for entry in &inputs.inventory.cases {
        let id = inputs::id(entry)?;
        let report = stage::read(root, entry)?;
        entries.push(id.into());
        if entry.artifact.is_some() {
            let path = format!("data/{id}.json");
            artifacts.insert(path.clone(), io::artifact(&root.join(path))?);
        }
        let path = format!("{id}/stage.json");
        artifacts.insert(path.clone(), io::artifact(&root.join(path))?);
        for endpoint in report.endpoints {
            let path = format!("{id}/{}.bin", name(endpoint.config.mode)?);
            if artifacts.insert(path, endpoint.checkpoint).is_some() {
                return Err("duplicate endpoint in global seal".into());
            }
        }
    }
    Ok(Seal {
        schema: 1,
        entries,
        artifacts,
    })
}
pub fn create(workspace: &Path, root: &Path) -> Result<()> {
    let inputs = inputs::load(root)?;
    let guard = Guard::new(workspace, root)?;
    let seal = collect(root, &inputs)?;
    io::json(&root.join("seal.json"), &seal, &guard)?;
    Ok(())
}
pub(crate) fn require(root: &Path, inputs: &inputs::Inputs) -> Result<io::Artifact> {
    let seal: Seal = io::read(&root.join("seal.json"))?;
    if seal != collect(root, inputs)? {
        return Err("global training seal changed or is incomplete".into());
    }
    io::artifact(&root.join("seal.json"))
}

//! The benchmark adapter owns the case identity, training split and provenance.
use crate::{
    Result,
    model::{Model, WIDTH},
    train,
};
use baseline::data::{Example, Task};
use bilinear::{
    experiment::{hash, write_json},
    guard,
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
};

pub const PROTOCOL: &str = include_str!("../protocol.json");
pub const SUITE: &str = include_str!("../../convex_stress/suite.json");

#[derive(Clone, Deserialize)]
pub struct Protocol {
    pub name: String,
    pub case: String,
    pub suite_sha256: String,
    pub data_seed: u32,
    pub seeds: Vec<u32>,
    pub modes: Vec<String>,
    pub context: usize,
    pub complex_width: usize,
    pub capacity: f64,
    pub initial_radius: f64,
    pub initial_key_scale: f64,
    pub updates: usize,
    pub batch: usize,
    pub embedding_lr: f64,
    pub routing_lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub query_step: f64,
    pub coordinates_per_table: usize,
}
pub fn protocol() -> Result<Protocol> {
    let p: Protocol = serde_json::from_str(PROTOCOL)?;
    if p.complex_width != WIDTH || p.suite_sha256 != hash(SUITE.as_bytes()) {
        return Err("architecture or original suite fingerprint changed".into());
    }
    Ok(p)
}
impl Protocol {
    pub fn initial(&self, vocabulary: usize, seed: u32) -> Result<Model> {
        Model::new(
            vocabulary,
            self.context,
            self.capacity,
            self.initial_radius,
            self.initial_key_scale,
            seed,
        )
    }
    pub fn cell(&self, mode: &str, seed: u32) -> Result<String> {
        if !self.modes.iter().any(|m| m == mode) || !self.seeds.contains(&seed) {
            return Err("cell outside frozen protocol".into());
        }
        Ok(format!("{mode}-{seed}"))
    }
    fn original(&self) -> Result<Value> {
        let suite: Value = serde_json::from_str(SUITE)?;
        if suite["data_seed"] != self.data_seed {
            return Err("original data seed mismatch".into());
        }
        suite["tasks"]
            .as_array()
            .ok_or("missing original cases")?
            .iter()
            .find(|row| row["name"] == self.case)
            .cloned()
            .ok_or_else(|| "case absent in original suite".into())
    }
    pub fn data(&self) -> Result<(Value, Task)> {
        let original = self.original()?;
        let task = baseline::data::build(&original)?;
        Ok((original, task))
    }
}
pub fn read(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

/// Build a new result in an adjacent directory and publish it as one rename.
/// A failed command keeps its partial evidence separate from a valid result.
pub fn publish(path: &Path, make: impl FnOnce(&Path) -> Result<()>) -> Result<()> {
    let parent = path.parent().ok_or("output has no parent")?;
    fs::create_dir_all(parent)?;
    if path.exists() {
        return Err(format!("output already exists: {}", path.display()).into());
    }
    let name = path
        .file_name()
        .ok_or("missing output name")?
        .to_string_lossy();
    let partial = parent.join(format!(".{name}.partial-{}", std::process::id()));
    fs::create_dir(&partial)?;
    make(&partial)?;
    fs::rename(partial, path)?;
    Ok(())
}
fn producer() -> Result<&'static str> {
    option_env!("RETRIEVAL_PRODUCER")
        .filter(|s| s.len() == 40 && s.bytes().all(|c| c.is_ascii_hexdigit()))
        .ok_or_else(|| "build from the frozen producing commit with the experiment runner".into())
}
pub fn fit(workspace: &Path, storage: &Path, root: &Path, mode: &str, seed: u32) -> Result<()> {
    guard::admit(1 << 30, workspace, storage)?;
    let study = Study::binding()?;
    let (_, task) = study.protocol.data()?;
    study.fit_cell(root, task.vocab, &task.splits["train"], mode, seed, || {
        guard::admit(1 << 30, workspace, storage).map(|_| ())
    })
}
pub struct Cell {
    pub name: String,
    pub seed: u32,
    pub model: Model,
}
/// Run identity, original data pins and the fixed training settings are owned by
/// the experiment harness. Only training examples and dimensions reach fitting.
pub struct Study {
    protocol: Protocol,
    identity: Value,
    fingerprints: Value,
}
impl Study {
    pub fn new(protocol: Protocol, definition: &str, fingerprints: Value) -> Result<Self> {
        let binary = hash(&fs::read(std::env::current_exe()?)?);
        let identity = json!({"producer":producer()?,"binary_sha256":binary,
            "protocol_sha256":hash(definition.as_bytes()),"suite_sha256":protocol.suite_sha256,
            "case":protocol.case});
        Ok(Self {
            protocol,
            identity,
            fingerprints,
        })
    }
    pub fn binding() -> Result<Self> {
        let p = protocol()?;
        let row = p.original()?;
        Self::new(p, PROTOCOL, row["data_fingerprints"].clone())
    }
    pub fn protocol(&self) -> &Protocol {
        &self.protocol
    }
    pub fn fit_cell(
        &self,
        root: &Path,
        vocabulary: usize,
        examples: &[Example],
        mode: &str,
        seed: u32,
        admit: impl FnMut() -> Result<()>,
    ) -> Result<()> {
        let p = &self.protocol;
        let name = p.cell(mode, seed)?;
        publish(&root.join(name), |out| {
            let mut model = p.initial(vocabulary, seed)?;
            let progress = train::fit(&mut model, examples, mode, seed, p, admit)?;
            let file = out.join("model.json");
            write_json(&file, &model)?;
            write_json(
                &out.join("receipt.json"),
                &json!({"identity":self.identity,"seed":seed,"mode":mode,
                "data_fingerprints":self.fingerprints,"vocabulary":vocabulary,"training":progress,
                "heldout_scored":false,"endpoint_sha256":hash(&fs::read(file)?)}),
            )
        })
    }
    fn checked_cells(&self, root: &Path) -> Result<(Value, Vec<Cell>)> {
        let p = &self.protocol;
        let mut cells = vec![];
        let mut pins = vec![];
        for mode in &p.modes {
            for &seed in &p.seeds {
                let name = p.cell(mode, seed)?;
                let dir = root.join(&name);
                let receipt_bytes = fs::read(dir.join("receipt.json"))?;
                let receipt: Value = serde_json::from_slice(&receipt_bytes)?;
                let bytes = fs::read(dir.join("model.json"))?;
                let endpoint_hash = hash(&bytes);
                if receipt["identity"] != self.identity
                    || receipt["seed"] != seed
                    || receipt["mode"] != *mode
                    || receipt["heldout_scored"] != false
                    || receipt["training"]["updates"] != p.updates
                    || receipt["data_fingerprints"] != self.fingerprints
                    || receipt["endpoint_sha256"] != endpoint_hash
                {
                    return Err(format!("invalid training receipt for {name}").into());
                }
                let model: Model = serde_json::from_slice(&bytes)?;
                model.validate()?;
                if model.context != p.context
                    || model.vocabulary
                        != receipt["vocabulary"].as_u64().ok_or("vocabulary")? as usize
                    || model.capacity != p.capacity
                {
                    return Err("endpoint architecture mismatch".into());
                }
                pins.push(json!({"cell":name,"receipt_sha256":hash(&receipt_bytes),"endpoint_sha256":endpoint_hash}));
                cells.push(Cell { name, seed, model });
            }
        }
        Ok((json!({"identity":self.identity,"cells":pins}), cells))
    }
    pub fn seal(&self, root: &Path) -> Result<()> {
        let (manifest, _) = self.checked_cells(root)?;
        publish(&root.join("sealed"), |dir| {
            write_json(&dir.join("manifest.json"), &manifest)
        })
    }
    pub fn load(&self, root: &Path) -> Result<Vec<Cell>> {
        let (manifest, cells) = self.checked_cells(root)?;
        if read(&root.join("sealed/manifest.json"))? != manifest {
            return Err("training seal changed".into());
        }
        Ok(cells)
    }
}
pub fn seal(root: &Path) -> Result<()> {
    Study::binding()?.seal(root)
}
pub fn load(root: &Path, p: &Protocol) -> Result<Vec<Cell>> {
    let row = p.original()?;
    Study::new(p.clone(), PROTOCOL, row["data_fingerprints"].clone())?.load(root)
}
pub fn raw_file(dir: &Path, name: &str, split: &str) -> PathBuf {
    dir.join(format!("{name}-{}.f64le", split.replace('/', "_")))
}

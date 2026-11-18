//! Freeze each final model and optimizer receipt and check both independent
//! builds in full before either build can score a held-out record.
use crate::{
    Result,
    data::{self, PROTOCOL, Protocol},
    model::Model,
    train,
};
use bilinear::{
    experiment::{hash, write_json},
    guard,
};
use learned_retrieval::cohort::{publish, read};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

pub struct Run<'a> {
    pub workspace: &'a Path,
    pub storage: &'a Path,
    pub output: &'a Path,
    pub arm: &'a str,
    pub protocol: Protocol,
    producer: &'static str,
}
impl<'a> Run<'a> {
    pub fn new(
        workspace: &'a Path,
        storage: &'a Path,
        output: &'a Path,
        arm: &'a str,
    ) -> Result<Self> {
        let producer = option_env!("RETRIEVAL_PRODUCER")
            .filter(|x| x.len() == 40 && x.bytes().all(|c| c.is_ascii_hexdigit()))
            .ok_or("build with the frozen producer commit")?;
        if !["original", "repeat"].contains(&arm)
            || fs::read_to_string(output.join("producer.commit"))?.trim() != producer
        {
            return Err("producer or independent arm mismatch".into());
        }
        let run = Self {
            workspace,
            storage,
            output,
            arm,
            protocol: data::protocol()?,
            producer,
        };
        let current = fs::canonicalize(std::env::current_exe()?)?;
        if current != fs::canonicalize(output.join(arm).join("composed-retrieval"))? {
            return Err("run from the frozen arm binary".into());
        }
        run.admit()?;
        Ok(run)
    }
    pub fn admit(&self) -> Result<()> {
        guard::admit(1 << 30, self.workspace, self.storage)?;
        Ok(())
    }
    pub fn root(&self, arm: &str) -> PathBuf {
        self.output.join(arm).join("run")
    }
    fn identity(&self, arm: &str) -> Result<Value> {
        Ok(
            json!({"producer":self.producer,"protocol_sha256":hash(data::PROTOCOL.as_bytes()),
            "suite_sha256":self.protocol.suite_sha256,"catalog_sha256":self.protocol.catalog_sha256,
            "binary_sha256":hash(&fs::read(self.output.join(arm).join("composed-retrieval"))?)}),
        )
    }
    pub fn fit(&self, id: &str, mode: &str, seed: u32) -> Result<()> {
        let p = &self.protocol;
        if !p.seeds.contains(&seed) || !p.modes.iter().any(|s| s == mode) {
            return Err("cell outside frozen protocol".into());
        }
        let (task, pins) = data::original(p, id)?;
        let s = &p.cases[id];
        let identity = self.identity(self.arm)?;
        let out = self.root(self.arm).join(id).join(format!("{mode}-{seed}"));
        publish(&out, |dir| {
            let mut m = Model::new(task.vocab, s.context, p, seed)?;
            let training = train::fit(
                &mut m,
                &task.splits["train"],
                s.batch,
                mode,
                seed,
                p,
                || self.admit(),
            )?;
            let path = dir.join("model.json");
            write_json(&path, &m)?;
            write_json(
                &dir.join("receipt.json"),
                &json!({"identity":identity,"case":id,"mode":mode,"seed":seed,
                "data_fingerprints":pins,"endpoint_sha256":hash(&fs::read(path)?),"training":training,"heldout_scored":false}),
            )
        })
    }
    fn inspect(&self, arm: &str) -> Result<(Value, Value)> {
        let identity = self.identity(arm)?;
        let root = self.root(arm);
        let p = &self.protocol;
        let mut pins = BTreeMap::new();
        let mut shared = BTreeMap::new();
        for (id, s) in &p.cases {
            let original_sha256 = match (&s.metadata_sha256, &s.source_fingerprints) {
                (Some(digest), _) => digest.clone(),
                (None, Some(pins)) => hash(&serde_json::to_vec(pins)?),
                (None, None) => hash(&serde_json::to_vec(&data::original(p, id)?.1)?),
            };
            for mode in &p.modes {
                for &seed in &p.seeds {
                    let cell = format!("{id}/{mode}-{seed}");
                    let path = root.join(&cell);
                    let receipt = fs::read(path.join("receipt.json"))?;
                    let r: Value = serde_json::from_slice(&receipt)?;
                    let bytes = fs::read(path.join("model.json"))?;
                    let model: Model = serde_json::from_slice(&bytes)?;
                    let layout = model.validate()?;
                    if p.initial_log_odds.is_some() {
                        let z = r["training"]["depth_log_odds"]
                            .as_f64()
                            .ok_or("missing depth log odds")?;
                        if !z.is_finite() || model.weights[layout.depth] != crate::model::sigmoid(z)
                        {
                            return Err(
                                "physical depth differs from sealed optimizer coordinate".into()
                            );
                        }
                    }
                    if p.coordinates_per_table.is_none() {
                        let queries = if mode == "forward" {
                            2 * model.weights.len() * p.updates
                        } else {
                            0
                        };
                        let training = &r["training"];
                        if training["min_visits"] != p.updates
                            || training["max_visits"] != p.updates
                            || training["forward_loss_queries"] != queries
                        {
                            return Err(format!(
                                "dense visits or loss query count differ in {arm}/{cell}"
                            )
                            .into());
                        }
                    }
                    if r["identity"] != identity
                        || r["heldout_scored"] != false
                        || r["case"] != *id
                        || r["mode"] != *mode
                        || r["seed"] != seed
                        || hash(&serde_json::to_vec(&r["data_fingerprints"])?) != original_sha256
                        || r["endpoint_sha256"] != hash(&bytes)
                        || r["training"]["updates"] != p.updates
                        || r["training"]["min_visits"].as_u64().unwrap_or(0) == 0
                        || model.capacity != p.capacity
                        || model.output_offset_bound != p.output_offset_bound
                        || model.learned_product_radii != p.learned_product_radii
                        || model.signed_profile != p.signed_profile
                        || model.predecessor_match != (p.predecessor_match && s.context > 2)
                        || model.context != s.context
                        || model.vocabulary != s.vocabulary
                    {
                        return Err(format!("invalid frozen cell {arm}/{cell}").into());
                    }
                    pins.insert(cell.clone(), hash(&receipt));
                    shared.insert(
                        cell,
                        json!({"weights":r["endpoint_sha256"],"training":r["training"]}),
                    );
                }
            }
        }
        Ok((json!({"identity":identity,"cells":pins}), json!(shared)))
    }
    pub fn seal(&self) -> Result<()> {
        let (manifest, _) = self.inspect(self.arm)?;
        publish(&self.root(self.arm).join("sealed"), |dir| {
            write_json(&dir.join("manifest.json"), &manifest)
        })
    }
    pub fn both_sealed(&self) -> Result<()> {
        let mut previous = None;
        for arm in ["original", "repeat"] {
            let (manifest, shared) = self.inspect(arm)?;
            if read(&self.root(arm).join("sealed/manifest.json"))? != manifest {
                return Err("seal mismatch".into());
            }
            if previous.as_ref().is_some_and(|p| p != &shared) {
                return Err("independent endpoints or optimizer evidence differ".into());
            }
            previous = Some(shared);
        }
        Ok(())
    }
}
pub fn protocol_hash() -> String {
    hash(PROTOCOL.as_bytes())
}

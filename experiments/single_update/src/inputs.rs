//! Immutable inputs and provenance, including the original compiled CUDA kernels.
use crate::{
    application::State,
    domain::{
        Result,
        data::{Data, Plan},
        model::{self, Arm},
    },
    infrastructure::{self, storage},
};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};
pub use storage::hash;

pub fn require(ok: bool, message: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(message.into()) }
}

#[derive(Default)]
pub struct Ledger(BTreeMap<PathBuf, String>);
impl Ledger {
    pub fn read(&mut self, path: &Path) -> Result<Vec<u8>> {
        let bytes = fs::read(path)?;
        let digest = hash(&bytes);
        if let Some(old) = self.0.insert(path.to_path_buf(), digest.clone()) {
            require(old == digest, "input changed during audit")?;
        }
        Ok(bytes)
    }
    pub fn json(&mut self, path: &Path) -> Result<Value> {
        Ok(serde_json::from_slice(&self.read(path)?)?)
    }
    pub fn verify(&self) -> Result<Value> {
        for (path, digest) in &self.0 {
            require(hash(&fs::read(path)?) == *digest, "original input changed")?;
        }
        Ok(json!({"unchanged":true,"files":self.0}))
    }
}

pub fn protocol(ledger: &mut Ledger, root: &Path) -> Result<(Value, String)> {
    let original = ledger.json(&root.join("protocol.json"))?;
    let ptx = hash(
        [
            infrastructure::ops::kernels::PTX,
            infrastructure::rotate::kernels::PTX,
            infrastructure::gemm::kernels::PTX,
            infrastructure::product::kernels::PTX,
        ]
        .concat()
        .as_bytes(),
    );
    require(
        original["ptx_sha256"] == ptx,
        &format!(
            "compiled PTX mismatch: expected {}, got {}",
            original["ptx_sha256"], ptx
        ),
    )?;
    require(
        original["source_sha256"] == env!("TRAINER_SOURCE_SHA256")
            && original["ptx_sha256"] == ptx
            && original["betas"] == json!([0.9, 0.98])
            && original["epsilon"] == 1e-8
            && original["learning_rate"] == 0.001
            && original["weight_decay"] == 1.
            && original["warmup_updates"] == 10
            && original["optimizer"] == "coordinate AdamW",
        "source/PTX/optimizer mismatch",
    )?;
    let digest = hash(&serde_json::to_vec(&original)?);
    Ok((original, digest))
}

pub struct Saved {
    pub state: State,
    pub metadata: Value,
    pub last_report: Value,
}
pub struct Case<'a> {
    pub arm: Arm,
    pub seed: u64,
    pub step: usize,
    pub data: &'a Data,
    pub plan: &'a Plan,
    pub protocol: &'a str,
}
pub fn checkpoint(ledger: &mut Ledger, root: &Path, case: &Case<'_>) -> Result<Saved> {
    let path = root.join(format!(
        "{}-seed{}/step-{:06}",
        case.arm.name(),
        case.seed,
        case.step
    ));
    let meta = ledger.json(&path.join("checkpoint.json"))?;
    let bytes = ledger.read(&path.join("state.bin"))?;
    let history = ledger.read(&path.join("history.json"))?;
    let id = &meta["identity"];
    let initial = model::initialize(case.arm, case.seed);
    let original = Plan::new(case.seed, 10000);
    require(
        case.plan.boundaries[..=10000] == original.boundaries
            && case.plan.order[..original.order.len()] == original.order,
        "extended sampler is not prefix stable",
    )?;
    require(
        meta["state_sha256"] == hash(&bytes) && meta["history_sha256"] == hash(&history),
        "checkpoint payload hash mismatch",
    )?;
    require(
        id["source_sha256"] == env!("TRAINER_SOURCE_SHA256")
            && id["protocol_sha256"] == case.protocol
            && id["arm"] == case.arm.name()
            && id["seed"] == case.seed
            && id["stored_parameters"] == case.arm.count()
            && id["independent_parameters"] == 143360,
        "checkpoint identity mismatch",
    )?;
    for (field, digest) in [
        ("train_sha256", hash(&storage::words(&case.data.train))),
        (
            "validation_sha256",
            hash(&storage::words(&case.data.validation)),
        ),
        ("order_sha256", hash(&storage::words(&original.order))),
        ("initial_sha256", hash(&storage::floats(&initial))),
        (
            "embedding_sha256",
            hash(&storage::floats(&initial[..model::E])),
        ),
    ] {
        require(id[field] == digest, field)?;
    }
    let (step, state) = storage::decode(&bytes, case.arm.count())?;
    require(
        step == case.step && meta["step"] == step,
        "wrong saved step",
    )?;
    require(
        model::valid(case.arm, &state.parameters) && state.second.iter().all(|&v| v >= 0.),
        "invalid saved parameters/moments",
    )?;
    if step == 0 {
        require(
            state.parameters == initial
                && state.first.iter().chain(&state.second).all(|&v| v == 0.),
            "initial state mismatch",
        )?;
    }
    let reports: Vec<Value> = serde_json::from_slice(&history)?;
    require(
        reports.len() == step / 100 + 1
            && reports
                .iter()
                .enumerate()
                .all(|(i, r)| r["step"] == i * 100),
        "history cadence mismatch",
    )?;
    Ok(Saved {
        state,
        metadata: meta,
        last_report: reports.last().unwrap().clone(),
    })
}

pub fn source_hash() -> String {
    hash(
        &[
            include_bytes!("main.rs").as_slice(),
            include_bytes!("inputs.rs"),
            include_bytes!("measure.rs"),
            include_bytes!("forcing.rs"),
            include_bytes!("margins.rs"),
            include_bytes!("drift.rs"),
            include_bytes!("probe.rs"),
            include_bytes!("query_step.rs"),
            include_bytes!("query_summary.rs"),
            include_bytes!("instrumentation.rs"),
            include_bytes!("../build.rs"),
            include_bytes!("../Cargo.toml"),
            include_bytes!("../Cargo.lock"),
            include_bytes!("../PROTOCOL.md"),
            include_bytes!("../MARGIN_PROTOCOL.md"),
            include_bytes!("../../residual_dynamics/src/algebra.rs"),
            include_bytes!("../../residual_dynamics/src/geometry.rs"),
            include_bytes!("../../residual_dynamics/src/gradient.rs"),
            include_bytes!("../../residual_dynamics/src/svd.rs"),
            include_bytes!("../../flat_approx/fixtures/reference.rs"),
        ]
        .concat(),
    )
}

//! Frozen experiment inputs and source identity. Task identity is confined here.
use crate::{
    Result,
    domain::Shape,
    model::Model,
    train::{Mode, Record},
};
use baseline::data::{Example, Task};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::Path;
pub const PROTOCOL: &str = include_str!("../protocol.json");
pub const SCALED_PROTOCOL: &str = include_str!("../scaled_protocol.json");
pub const SUITE: &str = include_str!("../../convex_stress/suite.json");

pub fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}

pub fn fingerprint() -> String {
    let mut hash = Sha256::new();
    for source in [
        include_bytes!("lib.rs").as_slice(),
        include_bytes!("domain.rs"),
        include_bytes!("algebra.rs"),
        include_bytes!("model.rs"),
        include_bytes!("train.rs"),
        include_bytes!("guard.rs"),
        include_bytes!("scaling.rs"),
        include_bytes!("experiment.rs"),
        include_bytes!("experiment/train_run.rs"),
        include_bytes!("experiment/score.rs"),
        include_bytes!("bin/stress.rs"),
        include_bytes!("../Cargo.toml"),
        include_bytes!("../Cargo.lock"),
        include_bytes!("../../convex_stress_minibatch/src/metrics.rs"),
        include_bytes!("../../convex_stress/src/vendor/data.rs"),
        include_bytes!("../../convex_stress/src/vendor/arithmetic_data.rs"),
        include_bytes!("../../convex_stress/src/vendor/random.rs"),
        include_bytes!("../../convex_stress/src/vendor/pcg.rs"),
        include_bytes!("../../convex_stress/src/vendor/blake.rs"),
        include_bytes!("../../convex_stress/src/channel.rs"),
        PROTOCOL.as_bytes(),
        SCALED_PROTOCOL.as_bytes(),
        SUITE.as_bytes(),
    ] {
        hash.update((source.len() as u64).to_le_bytes());
        hash.update(source);
    }
    hash.finalize().iter().map(|x| format!("{x:02x}")).collect()
}

pub fn case(name: &str) -> Result<(Value, Value, Task)> {
    case_in(name, PROTOCOL)
}

pub fn case_in(name: &str, protocol_text: &str) -> Result<(Value, Value, Task)> {
    let protocol: Value = serde_json::from_str(protocol_text)?;
    let config = protocol["cases"]
        .as_array()
        .ok_or("missing cohort")?
        .iter()
        .find(|c| c["name"] == name)
        .ok_or("case outside frozen cohort")?
        .clone();
    let suite: Value = serde_json::from_str(SUITE)?;
    let row = suite["tasks"]
        .as_array()
        .ok_or("missing original suite")?
        .iter()
        .find(|c| c["name"] == name)
        .ok_or("case outside original35")?
        .clone();
    let task = baseline::data::build(&row)?;
    Ok((config, row, task))
}

pub fn shape(task: &Task) -> Result<Shape> {
    // Architecture uses only original object count and declared input/rollout
    // lengths. No label value or task formula enters the model construction.
    let cap = task
        .splits
        .values()
        .flatten()
        .map(|e| {
            e.tokens
                .len()
                .max(e.prompt.len() + e.limit.saturating_sub(1))
        })
        .max()
        .ok_or("empty dataset")?;
    Shape::new(task.vocab, cap.saturating_sub(1))
}

pub fn mode(text: &str) -> Result<Mode> {
    match text {
        "mini" => Ok(Mode::Mini),
        "forward" => Ok(Mode::ForwardMini),
        _ => Err("unknown mode".into()),
    }
}

pub fn records<'a>(examples: &'a [Example], batch: &[usize]) -> Result<Vec<Record<'a>>> {
    let mut records = vec![];
    for &i in batch {
        let example = examples.get(i).ok_or("invalid record index")?;
        if example.tokens.len() != example.targets.len() {
            return Err("source target mask shape changed".into());
        }
        for (position, &target) in example.targets.iter().enumerate() {
            if target == -100 {
                continue;
            }
            records.push(Record {
                prefix: &example.tokens[..=position],
                target: usize::try_from(target)?,
            });
        }
    }
    if records.is_empty() {
        return Err("batch has no supervised records".into());
    }
    Ok(records)
}

pub struct Scorer<'a>(pub &'a Model);
impl baseline::metrics::Predictor for Scorer<'_> {
    fn vocab(&self) -> usize {
        self.0.joint.shape.objects
    }
    fn logits(&self, prefix: &[i64]) -> Vec<f64> {
        self.0
            .logits(prefix)
            .unwrap_or_else(|error| panic!("learned forward refused: {error}"))
    }
}

pub fn write_json(path: &Path, value: &impl serde::Serialize) -> Result<()> {
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    serde_json::to_writer(&mut file, value)?;
    writeln!(file)?;
    file.sync_all()?;
    Ok(())
}

pub mod score;
pub mod train_run;

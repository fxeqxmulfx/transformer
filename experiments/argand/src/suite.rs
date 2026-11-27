//! The frozen canonical GPT stress cases, their decoder shapes and records.
use crate::{
    Result,
    data::{self, Task},
    hash,
};
use serde_json::Value;
use std::{fs, io::Write, path::Path};

/// The decoder dimensions of one case.
#[derive(Clone)]
pub struct Shape {
    pub width: usize,
    pub heads: usize,
    pub layers: usize,
    pub vocab: usize,
    pub max_length: usize,
    pub max_batch: usize,
}

/// One of the 25 short and 10 long cases of the original suite, with its
/// splits rebuilt and checked against the suite's fingerprints.
pub fn load(name: &str) -> Result<(Value, Task)> {
    let suite: Value = serde_json::from_str(arithmetic::ORIGINAL35)?;
    let row = suite["tasks"]
        .as_array()
        .ok_or("missing tasks")?
        .iter()
        .find(|row| row["name"] == name)
        .filter(|row| matches!(row["tier"].as_str(), Some("short" | "long")))
        .ok_or("expected a case from the canonical GPT tiers")?
        .clone();
    let task = data::build(&row)?;
    Ok((row, task))
}

/// The case's width, heads and depth, with room for its longest sequence
/// and its training batch. Scoring uses the same batch: the scorer treats
/// examples independently and adds their losses in order, so the batch
/// changes neither the exact count nor the loss.
pub fn shape(row: &Value, task: &Task) -> Result<Shape> {
    let number = |key: &str| {
        row["case"][key]
            .as_u64()
            .map(|x| x as usize)
            .ok_or("missing shape")
    };
    let max_length = task
        .splits
        .values()
        .flatten()
        .map(|e| {
            e.tokens
                .len()
                .max(e.prompt.len() + e.limit.saturating_sub(1))
        })
        .max()
        .ok_or("empty task")?;
    Ok(Shape {
        width: number("width")?,
        heads: number("heads")?,
        layers: number("layers")?,
        vocab: task.vocab,
        max_length,
        max_batch: number("batch")?,
    })
}

/// The digest of the frozen suite.
pub fn suite_hash() -> String {
    hash(arithmetic::ORIGINAL35.as_bytes())
}

/// Writes `name` in `dir` atomically.
pub fn publish(dir: &Path, name: &str, bytes: &[u8]) -> Result<()> {
    let pending = dir.join(format!("{name}.pending"));
    let mut file = fs::File::create(&pending)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    fs::rename(pending, dir.join(name))?;
    fs::File::open(dir)?.sync_all()?;
    Ok(())
}

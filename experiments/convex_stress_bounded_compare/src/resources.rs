//! The later user limit is110GB RAM. Disk has no110GB quota: both filesystems
//! must fit original and repeated outputs. Insufficient space stops the run;
//! no existing artifact is removed. Checks include ongoing-write progress.
use crate::*;
use serde::{Deserialize, Serialize};
use std::{
    path::{Path, PathBuf},
    process::Command,
};

#[derive(Debug)]
pub struct ResourceStop(pub String);
impl std::fmt::Display for ResourceStop {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "resource stop: {}", self.0)
    }
}
impl std::error::Error for ResourceStop {}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Volume {
    pub source: String,
    pub available: u64,
}
pub fn volume(path: &Path) -> Result<Volume> {
    let output = Command::new("df")
        .args(["-B1", "--output=source,avail", "--"])
        .arg(path)
        .output()?;
    if !output.status.success() {
        return Err(ResourceStop("cannot check disk availability".into()).into());
    }
    let text = std::str::from_utf8(&output.stdout)?;
    let mut lines = text.lines().skip(1).filter(|s| !s.trim().is_empty());
    let mut fields = lines
        .next()
        .ok_or("missing disk availability")?
        .split_whitespace();
    let source = fields
        .next()
        .ok_or("missing filesystem source")?
        .to_string();
    let available = fields.next().ok_or("missing filesystem bytes")?.parse()?;
    if fields.next().is_some() || lines.next().is_some() {
        return Err("ambiguous disk availability".into());
    }
    Ok(Volume { source, available })
}

pub fn admit(
    workspace: &Volume,
    storage: &Volume,
    workspace_need: u64,
    storage_need: u64,
) -> Result<()> {
    let fits = if workspace.source == storage.source {
        workspace_need
            .checked_add(storage_need)
            .is_some_and(|n| n <= workspace.available.min(storage.available))
    } else {
        workspace_need <= workspace.available && storage_need <= storage.available
    };
    if !fits {
        return Err(ResourceStop(format!("both-disk admission failed: workspace {} available/{} needed; storage {} available/{} needed",
            workspace.available,workspace_need,storage.available,storage_need)).into());
    }
    Ok(())
}
fn used(path: &Path) -> Result<u64> {
    if !path.exists() {
        return Ok(0);
    }
    let meta = std::fs::symlink_metadata(path)?;
    if meta.file_type().is_symlink() {
        return Err("run artifact directory contains a symlink".into());
    }
    if meta.is_file() {
        return Ok(meta.len());
    }
    let mut total = 0u64;
    for entry in std::fs::read_dir(path)? {
        total = total
            .checked_add(used(&entry?.path())?)
            .ok_or("run-byte count overflow")?;
    }
    Ok(total)
}

#[derive(Clone, Debug)]
pub struct Guard {
    pub workspace: PathBuf,
    pub storage: PathBuf,
    pub run: PathBuf,
    /// Both runs for an original; the remaining one run for a repeat.
    pub output_budget: u64,
    pub workspace_reserve: u64,
    pub storage_reserve: u64,
}
impl Guard {
    pub fn check(&self) -> Result<()> {
        let written = used(&self.run)?;
        if written > self.output_budget {
            return Err(ResourceStop("run exceeds its admitted output budget".into()).into());
        }
        let need = (self.output_budget - written)
            .checked_add(self.storage_reserve)
            .ok_or("remaining disk forecast overflow")?;
        admit(
            &volume(&self.workspace)?,
            &volume(&self.storage)?,
            self.workspace_reserve,
            need,
        )
    }
    pub fn contains(&self, path: &Path) -> Result<()> {
        let parent = path
            .parent()
            .ok_or("artifact has no parent")?
            .canonicalize()?;
        if !parent.starts_with(self.run.canonicalize()?) {
            return Err("artifact escapes admitted run directory".into());
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaseBudget {
    pub model_bytes: u64,
    pub raw_bytes: u64,
    pub metadata_bytes: u64,
    pub total_bytes: u64,
}
pub fn case_budget(shape: joint::Shape, training: usize, validation: usize) -> Result<CaseBudget> {
    shape.admit(training)?;
    if validation == 0 || validation > 65536 {
        return Err("invalid original validation size".into());
    }
    let model = 3u128 * crate::tensor::length(shape)? as u128;
    // Every raw row carries all logits and exact maximizers, both inputs,
    // target, prediction and numerical diagnostics. It carries no task text
    // or program-state list. Each binary64 number occupies at most25 bytes.
    let raw = 3u128 * (training + validation) as u128 * (2048 + 48 * shape.classes) as u128;
    let metadata = 3u128 * (2_000_000 + 8 * shape.programs * shape.states) as u128;
    let total = model + raw + metadata;
    Ok(CaseBudget {
        model_bytes: u64::try_from(model)?,
        raw_bytes: u64::try_from(raw)?,
        metadata_bytes: u64::try_from(metadata)?,
        total_bytes: u64::try_from(total)?,
    })
}

//! Small native data preparation: bound reads/writes, both disks and total RSS.
use crate::{Result, hash};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    process::Command,
};
pub const FILE_LIMIT: usize = 32 * 1024 * 1024;
pub const RAM_LIMIT: u64 = 110_000_000_000;
pub const GROWTH_RESERVE: u64 = 4 * 1024 * 1024 * 1024;
pub const DISK_RESERVE: u64 = 2 * 1024 * 1024 * 1024;
pub struct Guard {
    pub workspace: PathBuf,
    pub storage: PathBuf,
}
pub fn admitted(workspace_free: u64, storage_free: u64, rss: u64) -> bool {
    workspace_free >= DISK_RESERVE
        && storage_free >= DISK_RESERVE
        && rss
            .checked_add(GROWTH_RESERVE)
            .is_some_and(|x| x <= RAM_LIMIT)
}
fn free(path: &Path) -> Result<u64> {
    let out = Command::new("df")
        .args(["-B1", "--output=avail", "--"])
        .arg(path)
        .output()?;
    if !out.status.success() {
        return Err("cannot check free disk space".into());
    }
    Ok(std::str::from_utf8(&out.stdout)?
        .lines()
        .nth(1)
        .ok_or("invalid df output")?
        .trim()
        .parse()?)
}
impl Guard {
    pub fn check(&self) -> Result<()> {
        let mut rss = 0u64;
        for process in fs::read_dir("/proc")? {
            let process = process?;
            if process
                .file_name()
                .to_string_lossy()
                .parse::<u32>()
                .is_err()
            {
                continue;
            }
            let Ok(status) = fs::read_to_string(process.path().join("status")) else {
                continue;
            };
            if let Some(line) = status.lines().find(|line| line.starts_with("VmRSS:")) {
                let kib: u64 = line
                    .split_whitespace()
                    .nth(1)
                    .ok_or("invalid VmRSS")?
                    .parse()?;
                rss = rss
                    .checked_add(kib.checked_mul(1024).ok_or("RSS overflow")?)
                    .ok_or("RSS overflow")?;
            }
        }
        if !admitted(free(&self.workspace)?, free(&self.storage)?, rss) {
            return Err(
                "stop: either disk lacks reserve or total RAM admission exceeds110GB".into(),
            );
        }
        Ok(())
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Artifact {
    pub bytes: usize,
    pub sha256: String,
}
pub fn read(path: &Path) -> Result<Vec<u8>> {
    use std::io::Read;
    let mut bytes = vec![];
    fs::File::open(path)?
        .take(FILE_LIMIT as u64 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > FILE_LIMIT {
        return Err("bounded data artifact is oversized".into());
    }
    Ok(bytes)
}
pub fn artifact(bytes: &[u8]) -> Artifact {
    Artifact {
        bytes: bytes.len(),
        sha256: hash(bytes),
    }
}
pub fn json(path: &Path, value: &impl Serialize, guard: &Guard) -> Result<Artifact> {
    let mut bytes = serde_json::to_vec(value)?;
    bytes.push(b'\n');
    if bytes.len() > FILE_LIMIT {
        return Err("data artifact exceeds32MiB admission".into());
    }
    guard.check()?;
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    guard.check()?;
    Ok(artifact(&bytes))
}

//! Both-disk admission for actual checkpoint writes. Full-cohort callers
//! supply their complete remaining forecast, including the independent repeat.
//! At least one extra checkpoint and 2/16 GiB workspace/storage slack are reserved.
use crate::Result;
use std::{
    fs,
    io::{BufWriter, Write},
    os::unix::fs::MetadataExt,
    path::{Path, PathBuf},
    process::Command,
};
const WORKSPACE_SLACK: u64 = 2 * 1024u64.pow(3);
const STORAGE_SLACK: u64 = 16 * 1024u64.pow(3);

pub fn sufficient(workspace: u64, storage: u64, same: bool, remaining: u64) -> Result<()> {
    let storage_need = STORAGE_SLACK
        .checked_add(remaining)
        .ok_or("disk forecast overflow")?;
    let workspace_need = if same {
        WORKSPACE_SLACK
            .checked_add(storage_need)
            .ok_or("shared-disk forecast overflow")?
    } else {
        WORKSPACE_SLACK
    };
    if workspace < workspace_need || storage < storage_need {
        return Err("insufficient workspace or storage disk space".into());
    }
    Ok(())
}
fn available(path: &Path) -> Result<u64> {
    let out = Command::new("df")
        .args(["-B1", "--output=avail", "--"])
        .arg(path)
        .output()?;
    if !out.status.success() {
        return Err("could not read filesystem free space".into());
    }
    let line = std::str::from_utf8(&out.stdout)?
        .lines()
        .nth(1)
        .ok_or("missing disk space output")?;
    Ok(line.trim().parse()?)
}
pub struct Guard {
    workspace: PathBuf,
    storage: PathBuf,
    same: bool,
    remaining: u64,
}
impl Guard {
    pub fn new(workspace: &Path, storage: &Path, remaining: u64) -> Result<Self> {
        let workspace = workspace.canonicalize()?;
        let storage = storage.canonicalize()?;
        let same = fs::metadata(&workspace)?.dev() == fs::metadata(&storage)?.dev();
        let guard = Self {
            workspace,
            storage,
            same,
            remaining,
        };
        guard.check()?;
        Ok(guard)
    }
    pub fn check(&self) -> Result<()> {
        sufficient(
            available(&self.workspace)?,
            available(&self.storage)?,
            self.same,
            self.remaining,
        )
    }
    pub fn writer(self, path: &Path, limit: u64) -> Result<Writer> {
        let parent = path
            .parent()
            .ok_or("checkpoint has no parent")?
            .canonicalize()?;
        if !parent.starts_with(&self.storage) {
            return Err("checkpoint must be inside admitted storage".into());
        }
        if fs::metadata(&parent)?.dev() != fs::metadata(&self.storage)?.dev() {
            return Err("checkpoint parent is on an unadmitted filesystem".into());
        }
        self.check()?;
        let file = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(path)?;
        Ok(Writer {
            file: BufWriter::with_capacity(65536, file),
            guard: self,
            limit,
            written: 0,
            next: 64 * 1024 * 1024,
        })
    }
}
pub struct Writer {
    file: BufWriter<fs::File>,
    guard: Guard,
    limit: u64,
    written: u64,
    next: u64,
}
fn io_error(e: Box<dyn std::error::Error>) -> std::io::Error {
    std::io::Error::other(e.to_string())
}
impl Write for Writer {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if buf.len() as u64 > self.limit.saturating_sub(self.written) {
            return Err(std::io::Error::other(
                "checkpoint exceeds admitted byte length",
            ));
        }
        let allowed = (self.next - self.written).min(buf.len() as u64) as usize;
        let n = self.file.write(&buf[..allowed])?;
        self.written += n as u64;
        self.guard.remaining = self.guard.remaining.saturating_sub(n as u64);
        if self.written >= self.next {
            self.flush()?;
            self.next = self.written.saturating_add(64 * 1024 * 1024);
        }
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.file.flush()?;
        self.guard.check().map_err(io_error)
    }
}
impl Writer {
    pub fn finish(mut self) -> Result<()> {
        self.flush()?;
        if self.written != self.limit {
            return Err("checkpoint ended before admitted byte length".into());
        }
        Ok(())
    }
}

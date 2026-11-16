//! Both disks and aggregate live RSS, with an independent production watchdog.
//! The small probe process has a4GiB address-space cap inside the user's110GB
//! aggregate RAM ceiling. Resource failures stop; existing evidence is retained.
use crate::Result;
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};
pub const PROCESS: u64 = model::input::PROCESS;
pub const RAM: u64 = 110_000_000_000;
pub const FILE_MAX: u64 = 1024 * 1024 * 1024;
pub const RUN_MAX: u64 = 16 * 1024 * 1024 * 1024;
pub const WORKSPACE_RESERVE: u64 = 2 * 1024 * 1024 * 1024;
pub const STORAGE_RESERVE: u64 = 16 * 1024 * 1024 * 1024;
#[derive(Clone)]
pub struct Guard {
    pub workspace: PathBuf,
    pub storage: PathBuf,
}
pub fn admitted(workspace: u64, storage: u64, same: bool, rss: u64) -> bool {
    let storage_need = 2 * RUN_MAX + STORAGE_RESERVE;
    let disks = if same {
        workspace.min(storage) >= WORKSPACE_RESERVE + storage_need
    } else {
        workspace >= WORKSPACE_RESERVE && storage >= storage_need
    };
    disks && rss.checked_add(PROCESS).is_some_and(|n| n <= RAM)
}
fn volume(path: &Path) -> Result<(String, u64)> {
    let out = Command::new("df")
        .args(["-B1", "--output=source,avail", "--"])
        .arg(path)
        .output()?;
    if !out.status.success() {
        return Err("cannot check disk availability".into());
    }
    let text = std::str::from_utf8(&out.stdout)?;
    let mut lines = text.lines().skip(1).filter(|s| !s.trim().is_empty());
    let mut fields = lines.next().ok_or("missing volume")?.split_whitespace();
    let source = fields.next().ok_or("missing filesystem")?.to_string();
    let free = fields.next().ok_or("missing free space")?.parse()?;
    if fields.next().is_some() || lines.next().is_some() {
        return Err("ambiguous volume".into());
    }
    Ok((source, free))
}
impl Guard {
    pub fn new(workspace: &Path, storage: &Path) -> Result<Self> {
        let out = Self {
            workspace: workspace.canonicalize()?,
            storage: storage.canonicalize()?,
        };
        out.check()?;
        Ok(out)
    }
    pub fn check(&self) -> Result<()> {
        let a = volume(&self.workspace)?;
        let b = volume(&self.storage)?;
        let mut rss = 0u64;
        for entry in fs::read_dir("/proc")? {
            let entry = entry?;
            if entry.file_name().to_string_lossy().parse::<u32>().is_err() {
                continue;
            }
            let Ok(text) = fs::read_to_string(entry.path().join("status")) else {
                continue;
            };
            if let Some(line) = text.lines().find(|line| line.starts_with("VmRSS:")) {
                let kib: u64 = line
                    .split_whitespace()
                    .nth(1)
                    .ok_or("invalid RSS")?
                    .parse()?;
                rss = rss
                    .checked_add(kib.checked_mul(1024).ok_or("RSS overflow")?)
                    .ok_or("RSS overflow")?;
            }
        }
        if !admitted(a.1, b.1, a.0 == b.0, rss) {
            return Err("resource stop: both disks or aggregate110GB RAM admission failed".into());
        }
        Ok(())
    }
    /// Only the CLI invokes this. The watchdog exits this process on shortage;
    /// it never signals another run, deletes output or attempts an automatic retry.
    pub fn watch(self) -> Result<()> {
        self.check()?;
        let pid = std::process::id().to_string();
        let cap = format!("--as={PROCESS}:{PROCESS}");
        if !Command::new("prlimit")
            .args(["--pid", &pid, &cap])
            .status()?
            .success()
        {
            return Err("cannot enforce probe address-space limit".into());
        }
        std::thread::Builder::new()
            .name("probe-resource-watch".into())
            .spawn(move || {
                loop {
                    std::thread::sleep(std::time::Duration::from_secs(2));
                    if let Err(error) = self.check() {
                        eprintln!("{error}");
                        std::process::exit(75);
                    }
                }
            })?;
        Ok(())
    }
}

//! Native aggregate-RAM and both-filesystem admission; no cleanup or fallback.
use crate::Result;
use std::{fs, path::Path, process::Command};

pub fn admit(reserved: u64, workspace: &Path, storage: &Path) -> Result<u64> {
    let mut rss = 0u64;
    for entry in fs::read_dir("/proc")? {
        let entry = entry?;
        if !entry
            .file_name()
            .to_string_lossy()
            .bytes()
            .all(|b| b.is_ascii_digit())
        {
            continue;
        }
        let Ok(status) = fs::read_to_string(entry.path().join("status")) else {
            continue;
        };
        for line in status.lines() {
            if let Some(value) = line.strip_prefix("VmRSS:") {
                rss = rss
                    .checked_add(
                        value
                            .split_whitespace()
                            .next()
                            .ok_or("invalid RSS")?
                            .parse::<u64>()?
                            .checked_mul(1024)
                            .ok_or("RSS overflow")?,
                    )
                    .ok_or("aggregate RSS overflow")?;
            }
        }
    }
    if rss
        .checked_add(reserved)
        .ok_or("RAM reservation overflow")?
        > 110_000_000_000
    {
        return Err(
            format!("aggregate RAM admission refused: RSS={rss}, reservation={reserved}").into(),
        );
    }
    for (path, minimum) in [(workspace, 2_147_483_648u64), (storage, 17_179_869_184)] {
        let output = Command::new("df")
            .args(["-B1", "--output=avail"])
            .arg(path)
            .output()?;
        if !output.status.success() {
            return Err("both-disk availability check failed".into());
        }
        let text = String::from_utf8(output.stdout)?;
        let free = text
            .split_whitespace()
            .last()
            .ok_or("missing disk availability")?
            .parse::<u64>()?;
        if free < minimum {
            return Err(format!(
                "disk admission refused: {} has {free} bytes",
                path.display()
            )
            .into());
        }
    }
    Ok(rss)
}

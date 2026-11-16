//! A new large run must not overlap existing experiment pipelines. Their
//! wrappers are checked too, so the gap between child phases is not mistaken
//! for completion. Other process RSS and an additional growth reserve are
//! subtracted from110GB before computing the model's process allowance.
use super::*;
use std::{collections::BTreeSet, fs, io::ErrorKind};
const GROWTH_RESERVE: u64 = 2 * 1024 * 1024 * 1024;
fn field(status: &str, name: &str) -> Result<u64> {
    status
        .lines()
        .find_map(|line| line.strip_prefix(name))
        .and_then(|v| v.split_whitespace().next())
        .ok_or("missing process status field")?
        .parse()
        .map_err(Into::into)
}
fn experiment(args: &[&str]) -> bool {
    let binary = args
        .first()
        .and_then(|s| Path::new(s).file_name())
        .and_then(|s| s.to_str())
        .unwrap_or("");
    (binary.starts_with("convex-stress-")
        && args
            .get(1)
            .is_some_and(|p| ["stage", "score", "replay"].contains(p)))
        || (["bash", "sh"].contains(&binary)
            && args
                .iter()
                .any(|s| s.contains("/experiments/convex_stress_") && s.ends_with("/run.sh")))
}
fn allowance(other: u64) -> Result<u64> {
    let cap = RAM_LIMIT_BYTES
        .checked_sub(other)
        .and_then(|v| v.checked_sub(GROWTH_RESERVE))
        .ok_or_else(|| ResourceStop("other processes exhaust the shared RAM limit".into()))?;
    if cap <= joint::admission::MIN_RUNTIME_BYTES {
        return Err(ResourceStop("insufficient shared RAM for model runtime".into()).into());
    }
    Ok(cap)
}
pub fn process_limit() -> Result<u64> {
    let own = std::process::id() as u64;
    let mut ancestors = BTreeSet::new();
    let mut pid = own;
    while pid != 0 && ancestors.insert(pid) {
        pid = field(&fs::read_to_string(format!("/proc/{pid}/status"))?, "PPid:")?;
    }
    let mut other = 0u64;
    for entry in fs::read_dir("/proc")? {
        let entry = entry?;
        let Ok(pid) = entry.file_name().to_string_lossy().parse::<u64>() else {
            continue;
        };
        if pid == own {
            continue;
        }
        let status = match fs::read_to_string(entry.path().join("status")) {
            Ok(s) => s,
            Err(e) if e.kind() == ErrorKind::NotFound => continue,
            Err(e) => return Err(e.into()),
        };
        // Kernel threads and terminated processes do not have a VmRSS field.
        if status.lines().any(|s| s.starts_with("VmRSS:")) {
            other = other
                .checked_add(
                    field(&status, "VmRSS:")?
                        .checked_mul(1024)
                        .ok_or("process RSS overflow")?,
                )
                .ok_or("aggregate RSS overflow")?;
        }
        if ancestors.contains(&pid) {
            continue;
        }
        let bytes = match fs::read(entry.path().join("cmdline")) {
            Ok(b) => b,
            Err(e) if e.kind() == ErrorKind::NotFound => continue,
            Err(e) => return Err(e.into()),
        };
        let text = String::from_utf8_lossy(&bytes);
        let args: Vec<_> = text.split('\0').filter(|s| !s.is_empty()).collect();
        if experiment(&args) {
            return Err(ResourceStop(format!("existing experiment pipeline PID{pid} is still live; leave it unchanged and wait before starting a new large run")).into());
        }
    }
    let limit = allowance(other)?;
    eprintln!(
        "Shared RAM: other process RSS={other} bytes; growth reserve={GROWTH_RESERVE}; admitted process={limit}; global limit={RAM_LIMIT_BYTES}"
    );
    Ok(limit)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn live_wrappers_and_children_are_detected_and_other_ram_is_not_free_space() {
        assert!(experiment(&[
            "/tmp/convex-stress-reduced-compare",
            "stage",
            "/root"
        ]));
        assert!(experiment(&[
            "bash",
            "/freeze/source/experiments/convex_stress_bounded_compare/run.sh"
        ]));
        assert!(!experiment(&[
            "cat",
            "/freeze/source/experiments/convex_stress_bounded_compare/run.sh"
        ]));
        assert!(!experiment(&[
            "/tmp/convex-stress-compact-compare",
            "forecast"
        ]));
        assert_eq!(
            allowance(20_000_000_000).unwrap(),
            RAM_LIMIT_BYTES - 20_000_000_000 - GROWTH_RESERVE
        );
        assert!(allowance(RAM_LIMIT_BYTES).is_err());
        assert!(allowance(RAM_LIMIT_BYTES - GROWTH_RESERVE).is_err());
    }
}

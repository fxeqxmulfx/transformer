//! The actual shell runner retains pre-start disk refusals and later outcomes.
use super::*;
use std::{os::unix::fs::PermissionsExt, process::Command};
fn executable(path: &Path, body: &str) -> Result<()> {
    fs::write(path, body)?;
    fs::set_permissions(path, fs::Permissions::from_mode(0o755))?;
    Ok(())
}
fn setup(temp: &Temp, low: bool) -> Result<(PathBuf, PathBuf, String)> {
    let bin = temp.0.join("bin");
    fs::create_dir(&bin)?;
    executable(
        &bin.join("agent-kanban"),
        "#!/bin/sh\nprintf '#%s high done synthetic-fixture\\n' \"$2\"\n",
    )?;
    let disk = if low {
        "#!/bin/sh\nprintf 'Filesystem 1024-blocks Used Available Capacity Mounted\\nfixture 100 99 1 99%% /fixture\\n'\n".to_string()
    } else {
        format!(
            "#!/bin/sh\nfile='{}'\nn=0\nif test -f \"$file\"; then n=$(cat \"$file\"); fi\nn=$((n+1))\nprintf '%s\\n' \"$n\" > \"$file\"\nfree=999999999\nif test \"$n\" = 2; then free=1; fi\nprintf 'Filesystem 1024-blocks Used Available Capacity Mounted\\nfixture 999999999 0 %s 0%% /fixture\\n' \"$free\"\n",
            temp.0.join("disk-count").display()
        )
    };
    executable(&bin.join("df"), &disk)?;
    let binary = bin.join("fixture-model");
    executable(
        &binary,
        r#"#!/bin/sh
set -eu
case "$1" in
 prepare) mkdir "$3"; printf '{"tasks":["first","second"]}\n' > "$3/inventory.json" ;;
 run-case) printf 'started\n' > "$2/$3.started" ;;
 replay-case) printf '{"synthetic_replay":true}\n' ;;
 summarize) printf '{"synthetic_summary":true}\n' ;;
 *) exit 2 ;;
esac
"#,
    )?;
    let request = temp.0.join("request.json");
    fs::write(&request, "{}\n")?;
    let path = format!("{}:{}", bin.display(), std::env::var("PATH")?);
    Ok((binary, request, path))
}
#[test]
fn disk_pressure_retains_exit_125_and_runs_later_planned_cases() -> Result<()> {
    let temp = Temp::new()?;
    let (binary, request, path) = setup(&temp, false)?;
    let destination = temp.0.join("results");
    let output = Command::new("bash")
        .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/run_original.sh"))
        .arg(request)
        .arg(&destination)
        .env("PATH", path)
        .env("SHARED_COMPARE_BINARY", binary)
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        fs::read_to_string(destination.join("first.run.exit"))?.trim(),
        "125"
    );
    assert!(!destination.join("first.started").exists());
    assert!(destination.join("second.started").exists());
    assert_eq!(
        fs::read_to_string(destination.join("second.replay.exit"))?.trim(),
        "0"
    );
    assert!(destination.join("first.run.resources").exists());
    assert!(destination.join("summary.json").exists());
    Ok(())
}
#[test]
fn initial_disk_refusal_creates_no_run_or_partial_prepare() -> Result<()> {
    let temp = Temp::new()?;
    let (binary, request, path) = setup(&temp, true)?;
    let destination = temp.0.join("results");
    let output = Command::new("bash")
        .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/run_original.sh"))
        .arg(request)
        .arg(&destination)
        .env("PATH", path)
        .env("SHARED_COMPARE_BINARY", binary)
        .output()?;
    assert!(!output.status.success());
    assert!(!destination.exists());
    assert!(!temp.0.join("results-prepare.resources").exists());
    Ok(())
}

use super::*;
use std::{os::unix::fs::PermissionsExt, process::Command};

struct Cleanup(PathBuf);
impl Drop for Cleanup {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn executable(path: &Path, script: &str) -> Result<()> {
    fs::write(path, script)?;
    fs::set_permissions(path, fs::Permissions::from_mode(0o755))?;
    Ok(())
}

#[test]
fn disk_refusal_retains_the_case_then_continues_and_initial_refusal_stops() -> Result<()> {
    let temp = Cleanup(
        std::env::temp_dir().join(format!("kernel-comparison-runner-{}", std::process::id())),
    );
    let bin = temp.0.join("bin");
    fs::create_dir_all(&bin)?;
    executable(
        &bin.join("agent-kanban"),
        "#!/bin/sh\nprintf '#%s high done fixture\\n' \"$2\"\n",
    )?;
    executable(
        &bin.join("df"),
        r#"#!/bin/sh
count=0
if test -f "$DISK_FIXTURE_ROOT/count"; then count=$(cat "$DISK_FIXTURE_ROOT/count"); fi
count=$((count+1))
printf '%s\n' "$count" > "$DISK_FIXTURE_ROOT/count"
available=33554432
if test "$count" = 2 || test "$DISK_FIXTURE_INITIAL_LOW" = 1; then available=0; fi
printf 'Filesystem 1024-blocks Used Available Capacity Mounted\nfixture 33554432 0 %s 0%% /\n' "$available"
"#,
    )?;
    let binary = bin.join("comparison-fixture");
    executable(
        &binary,
        r#"#!/bin/sh
set -eu
case "$1" in
  prepare) mkdir "$3"; printf '{"tasks":["first","second"]}\n' > "$3/inventory.json";;
  run-case) printf 'actual fixture metric process\n' > "$2/$3.started";;
  replay-case) printf '{}\n';;
  summarize) printf '{}\n';;
  *) exit 2;;
esac
"#,
    )?;
    let run = |destination: &Path, low: &str| {
        Command::new("bash")
            .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/run_original.sh"))
            .arg("fixture-request")
            .arg(destination)
            .env(
                "PATH",
                format!("{}:{}", bin.display(), std::env::var("PATH").unwrap()),
            )
            .env("DISK_FIXTURE_ROOT", &temp.0)
            .env("DISK_FIXTURE_INITIAL_LOW", low)
            .env("READOUT_COMPARE_BINARY", &binary)
            .output()
    };
    let root = temp.0.join("comparison");
    let out = run(&root, "0")?;
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(fs::read_to_string(root.join("first.run.exit"))?, "125\n");
    assert!(fs::read_to_string(root.join("first.run.resources"))?.contains("zero new evaluation"));
    assert!(!root.join("first.started").exists());
    assert_eq!(fs::read_to_string(root.join("second.run.exit"))?, "0\n");
    assert_eq!(fs::read_to_string(root.join("second.replay.exit"))?, "0\n");
    assert!(root.join("second.started").exists());
    assert_eq!(read::<Value>(&root.join("summary.json"))?, json!({}));
    let refused = temp.0.join("initial-refusal");
    let out = run(&refused, "1")?;
    assert!(!out.status.success());
    assert!(!refused.exists());
    Ok(())
}

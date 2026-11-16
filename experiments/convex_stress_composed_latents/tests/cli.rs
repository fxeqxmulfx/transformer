//! Exercise the actual executable boundary, guarded report output and strict
//! training-only schema; library tests alone do not cover dispatch or file I/O.
use std::{fs, process::Command};

#[test]
fn actual_probe_keeps_the_complete_partition_and_refuses_extra_inputs_or_overwrites() {
    let root = std::env::temp_dir().join(format!("composed-latent-cli-{}", std::process::id()));
    fs::create_dir(&root).unwrap();
    let input = root.join("training.json");
    let output = root.join("report.json");
    let refused = root.join("refused.json");
    fs::write(
        &input,
        br#"{"vocabulary":3,"rows":[[0,0,0],[0,1,1],[1,0,2]]}"#,
    )
    .unwrap();
    let run = |target: &std::path::Path| {
        Command::new(env!("CARGO_BIN_EXE_convex-stress-composed-latents"))
            .arg("probe")
            .arg(&input)
            .arg(target)
            .arg(&root)
            .arg(&root)
            .output()
            .unwrap()
    };
    let result = run(&output);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let original = fs::read(&output).unwrap();
    let report: serde_json::Value = serde_json::from_slice(&original).unwrap();
    let selected = &report["selection"];
    assert_eq!(
        selected["retained"].as_array().unwrap().len()
            + selected["conflicts"].as_array().unwrap().len(),
        convex_stress_composed_latents::PROGRAMS
    );
    for name in [
        "expanded_training_records_allocated",
        "heldout_pools_read",
        "accuracy_measurements",
        "training_updates",
    ] {
        assert_eq!(report[name], 0);
    }
    assert!(!run(&output).status.success());
    assert_eq!(fs::read(&output).unwrap(), original);
    fs::write(
        &input,
        br#"{"vocabulary":3,"rows":[[0,0,0]],"heldout":[[1,1,2]]}"#,
    )
    .unwrap();
    assert!(!run(&refused).status.success());
    assert!(!refused.exists());
    fs::remove_file(input).unwrap();
    fs::remove_file(output).unwrap();
    fs::remove_dir(root).unwrap();
}

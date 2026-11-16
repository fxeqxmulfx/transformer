use super::*;
use std::time::{SystemTime, UNIX_EPOCH};

fn inputs() -> Vec<Input> {
    vec![
        Input {
            tokens: vec![1],
            target: 1,
        },
        Input {
            tokens: vec![0],
            target: 0,
        },
        Input {
            tokens: vec![1],
            target: 1,
        },
    ]
}
fn config() -> Config {
    Config::new(&"a".repeat(40), &"b".repeat(64)).unwrap()
}
fn temporary() -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!(
        "learned-sequence-pilot-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir(&path).unwrap();
    path
}

#[test]
fn mini_trace_roundtrip_keeps_all_coordinates_certificates_and_exact_ce_bits() {
    let old = model().unwrap();
    let batch = records(&inputs()[..2]).unwrap();
    let update = old
        .mini_step(&batch, Some(&rename()), &ratio(1, 32), mini_limits())
        .unwrap();
    let trace = Step::mini(
        0,
        0,
        0,
        2,
        &update,
        mean_loss(&update.projected.model, &batch).unwrap(),
    );
    let encoded = serde_json::to_vec(&trace).unwrap();
    let mut decoded: Step = serde_json::from_slice(&encoded).unwrap();
    let next = decoded.replay(&old, &batch).unwrap();
    assert_eq!(next.parameters(), update.projected.model.parameters());
    decoded.final_loss ^= 1;
    assert!(decoded.replay(&old, &batch).is_err());
    let mut unknown: serde_json::Value = serde_json::from_slice(&encoded).unwrap();
    unknown["omitted_classes"] = serde_json::json!(1);
    assert!(serde_json::from_value::<Step>(unknown).is_err());
}

#[test]
fn forward_trace_roundtrip_replays_every_persisted_probe_and_original_object_state() {
    let old = model().unwrap();
    let batch = records(&inputs()[2..]).unwrap();
    let update = old
        .forward_mini_step(&batch, Some(&rename()), &forward_config(), forward_limits())
        .unwrap();
    let trace = Step::forward(1, 0, 2, 1, &update);
    let encoded = serde_json::to_vec(&trace).unwrap();
    let decoded: Step = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(
        decoded.replay(&old, &batch).unwrap().parameters(),
        update.projected.model.parameters()
    );
    let mut forged: serde_json::Value = serde_json::from_slice(&encoded).unwrap();
    forged["cells"][0]["probes"][0]["plus"] = serde_json::json!(0);
    let forged: Step = serde_json::from_value(forged).unwrap();
    assert!(forged.replay(&old, &batch).is_err());
}

#[test]
fn scoring_refuses_an_absent_training_seal_before_opening_invalid_evaluation_data() {
    let out = temporary();
    let evaluation = out.with_extension("invalid-evaluation.json");
    fs::write(&evaluation, b"invalid evaluation JSON").unwrap();
    let error = score(&out, &evaluation).unwrap_err();
    assert!(error.downcast_ref::<std::io::Error>().is_some());
    assert!(!out.join("evaluation.json").exists());
    assert!(!out.join("score.json").exists());
}

#[test]
fn sealed_file_inventory_cannot_replace_complete_indexed_training_replay() {
    let out = temporary();
    save(&out.join("config.json"), &config()).unwrap();
    save(&out.join("training.json"), &inputs()).unwrap();
    let old = model().unwrap();
    let batch = records(&inputs()[..2]).unwrap();
    let update = old
        .mini_step(&batch, Some(&rename()), &ratio(1, 32), mini_limits())
        .unwrap();
    let trace = Step::mini(
        0,
        0,
        0,
        2,
        &update,
        mean_loss(&update.projected.model, &batch).unwrap(),
    );
    for mode in MODES {
        for i in 0..16 {
            save(&out.join(step_name(mode, i)), &trace).unwrap();
        }
    }
    let mut files = BTreeMap::new();
    for name in expected_files(3) {
        files.insert(name.clone(), hash(&out.join(name)).unwrap());
    }
    let seal = Seal {
        version: 1,
        binary_sha: "c".repeat(64),
        records: 3,
        updates_per_mode: 16,
        heldout_pools_scored: 0,
        files,
        endpoints: MODES
            .into_iter()
            .map(|m| (m.into(), step_name(m, 15)))
            .collect(),
    };
    save(&out.join("training-seal.json"), &seal).unwrap();
    check_seal(&out).unwrap();
    assert!(
        replay_training(&out)
            .unwrap_err()
            .to_string()
            .contains("indexed minibatch")
    );
    fs::write(out.join(step_name("mini", 0)), b"corrupted").unwrap();
    assert!(check_seal(&out).is_err());
}

#[test]
fn original_data_identity_and_complete_mode_inventory_are_explicit() {
    assert_eq!(expected_files(3).len(), 34);
    assert_eq!(records(&inputs()).unwrap().len(), 3);
    assert!(records(&[]).is_err());
    assert!(
        records(&[Input {
            tokens: vec![3],
            target: 0
        }])
        .is_err()
    );
    assert!(
        records(&[Input {
            tokens: vec![0, 1],
            target: 0
        }])
        .is_err()
    );
    assert!(
        records(&[Input {
            tokens: vec![0],
            target: 3
        }])
        .is_err()
    );
    assert!(Config::new("invalid", &"b".repeat(64)).is_err());
}

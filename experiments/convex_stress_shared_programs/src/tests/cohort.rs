use super::*;
use serde_json::{Value, json};
use std::collections::BTreeMap;

fn row(tokens: &[usize], label: usize) -> data::Example {
    let mut targets = vec![-100; tokens.len()];
    *targets.last_mut().unwrap() = label as i64;
    let changes = targets.iter().map(|&v| v >= 0).collect();
    data::Example {
        task: "fixture".into(),
        tokens: tokens.iter().map(|&v| v as i64).collect(),
        targets,
        changes,
        prompt: vec![],
        answer: vec![],
        limit: 0,
        field: None,
        row_index: None,
    }
}
fn dataset() -> dataset::Dataset {
    let input = fixture();
    let train: Vec<_> = input
        .observations
        .iter()
        .map(|s| row(&s.tokens, s.label))
        .collect();
    let test = vec![
        row(&[1, 4, 7, 2, 5, 4], 7),
        row(&[1, 2, 4, 2, 5, 2], 4),
        row(&[1, 7, 2], 0),
    ];
    dataset::Dataset {
        name: "fixture".into(),
        task: data::Task {
            vocab: 8,
            selection: "validation".into(),
            splits: [
                ("train".into(), train),
                ("validation".into(), test.clone()),
                ("test".into(), test),
                ("test_long".into(), vec![row(&[0, 0, 1, 4, 7, 2, 5, 4], 7)]),
            ]
            .into(),
        },
        original_fingerprints: BTreeMap::new(),
        tensor_fingerprints: BTreeMap::new(),
        empty_novel_splits: vec![],
    }
}

#[test]
fn training_boundary_excludes_identity_and_all_heldout_labels() -> Result<()> {
    let mut data = dataset();
    let original = runner::construct(8, &data.task.splits["train"], json!({"name":"original"}))?;
    data.name = "different".into();
    for (split, rows) in &mut data.task.splits {
        for row in rows {
            row.task = "changed-name".into();
            row.changes.fill(false);
            if split != "train" {
                row.tokens.fill(0);
                row.targets.fill(0);
            }
        }
    }
    let changed = runner::construct(8, &data.task.splits["train"], json!({"name":"changed"}))?;
    assert_ne!(changed.dataset, original.dataset);
    assert_eq!(changed.input, original.input);
    assert_eq!(changed.model, original.model);
    assert_eq!(changed.error, original.error);
    Ok(())
}

#[test]
fn every_mode_retains_predictions_unseen_labels_ambiguity_and_missing_defaults() -> Result<()> {
    let data = dataset();
    let stage = runner::construct(8, &data.task.splits["train"], data.metadata())?;
    let report = runner::measurements(&stage, &data)?;
    assert_eq!(report["modes"].as_array().unwrap().len(), 3);
    for endpoint in report["modes"].as_array().unwrap() {
        assert!(endpoint["error"].is_null());
        let measurement = &endpoint["measurement"];
        assert_eq!(measurement["target_realizability_certified"], false);
        let test = &measurement["splits"]["test"];
        assert_eq!(test["predictions"].as_array().unwrap().len(), 3);
        assert_eq!(test["consensus_targets"], 2);
        assert_eq!(test["ambiguous_targets"], 1);
        assert_eq!(test["roundoff_sensitive_targets"], 1);
        assert_eq!(test["any_missing_targets"], 1);
        assert_eq!(test["unseen_label_targets"], 2);
        assert_eq!(test["unseen_label_correct"], 2);
        assert_eq!(test["predictions"][0]["prediction"], 7);
        assert_eq!(measurement["splits"]["test_long"]["goals"]["both99"], true);
        assert!(test["independent_complex_max_difference"].as_f64().unwrap() < 1e-10);
    }
    let restored: runner::Stage = serde_json::from_slice(&serde_json::to_vec(&stage)?)?;
    binding::check(&restored.input, restored.model.as_ref().unwrap())?;
    assert!(runner::artifacts::same_json(
        &report,
        &runner::measurements(&restored, &data)?
    )?);
    Ok(())
}

#[test]
fn all_three_refusals_goals_and_immutable_artifacts_remain_distinct() -> Result<()> {
    let mut data = dataset();
    data.task.splits.get_mut("train").unwrap()[24].targets[5] = 7;
    let stage = runner::construct(8, &data.task.splits["train"], data.metadata())?;
    binding::check(&stage.input, stage.model.as_ref().unwrap())?;
    let report = runner::measurements(&stage, &data)?;
    assert_eq!(report["modes"].as_array().unwrap().len(), 3);
    for mode in report["modes"].as_array().unwrap() {
        assert!(mode["error"].is_string());
        assert!(mode["measurement"].is_null());
        assert!(mode["primary"].is_null());
        assert_eq!(mode["all_heldout99"], false);
    }
    let goals =
        sequences::adapter::goals(&json!({"token_accuracy":0.995,"sequence_accuracy":0.96}));
    assert_eq!(
        goals,
        json!({"raw99":true,"exact99":false,"both99":false,"original_success":true})
    );
    let dir = std::env::temp_dir().join(format!("shared-copy-fixture-{}", std::process::id()));
    std::fs::create_dir(&dir)?;
    let path = dir.join("state.json");
    runner::artifacts::write(&path, &report)?;
    assert!(runner::artifacts::write(&path, &json!({"replaced":true})).is_err());
    assert_eq!(runner::artifacts::read::<Value>(&path)?, report);
    assert!(runner::score(&dir, "suite-mqar").is_err());
    std::fs::remove_dir_all(dir)?;
    Ok(())
}

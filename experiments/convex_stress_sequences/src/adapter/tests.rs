use super::*;
use crate::{dataset::Dataset, runner, wire};

fn fixture() -> Dataset {
    let rows: Vec<_> = (0..67)
        .map(|i| {
            wire::example(
                "fixture",
                vec![1, if i % 2 == 0 { 0 } else { 2 }, 3],
                vec![-100, -100, if i % 2 == 0 { 1 } else { 2 }],
            )
        })
        .collect();
    Dataset {
        name: "fixture".into(),
        task: data::Task {
            vocab: 4,
            selection: "validation".into(),
            splits: [
                ("train".into(), rows.clone()),
                ("validation".into(), rows.clone()),
                ("test".into(), rows),
            ]
            .into(),
        },
        original_fingerprints: BTreeMap::new(),
        tensor_fingerprints: BTreeMap::new(),
        empty_novel_splits: vec![],
    }
}

#[test]
fn training_boundary_ignores_identity_and_heldout_answers() -> Result<()> {
    let data = fixture();
    let initial = request(data.task.vocab, &data.task.splits["train"])?;
    let mut changed = fixture();
    for (split, rows) in &mut changed.task.splits {
        for row in rows {
            row.task = "different-name".into();
            row.changes.fill(false);
            if split != "train" {
                row.targets.fill(0);
                row.tokens.fill(0);
            }
        }
    }
    assert_eq!(
        initial,
        request(changed.task.vocab, &changed.task.splits["train"])?
    );
    assert_eq!(initial.discovery.input.context, 3);
    assert_eq!(initial.discovery.input.observations.len(), 67);
    assert_eq!(
        initial.discovery.input.observations[0].tokens,
        vec![1, 0, 3]
    );
    let mut bad = data.task.splits["train"].clone();
    bad[0].targets[0] = -1;
    assert!(request(4, &bad).is_err());
    bad[0].targets[0] = -100;
    bad[0].tokens[0] = -1;
    assert!(request(4, &bad).is_err());
    Ok(())
}

#[test]
fn actual_modes_replay_complete_complex_predictions_and_cyclic_minibatches() -> Result<()> {
    let data = fixture();
    let stage = runner::construct(4, &data.task.splits["train"], data.metadata())?;
    let report = stage.model.as_ref().unwrap();
    binding::replay(&stage.request, report)?;
    let expected = [(16, 16 * 67), (256, 256 * 64), (0, 0)];
    for (endpoint, (batches, examples)) in report.endpoints.iter().zip(expected) {
        let training = endpoint.training.as_ref().unwrap();
        assert_eq!(training.work.gradient_batches, batches);
        assert_eq!(training.work.gradient_examples, examples);
        assert_eq!(training.steps.len(), batches);
        let value = evaluate(&training.model, &data.task.splits)?;
        assert_eq!(value["splits"]["test"]["goals"]["both99"], true);
        assert_eq!(value["splits"]["test"]["metrics"]["correct"], 67);
        assert_eq!(value["splits"]["test"]["known_errors"], 0);
        assert_eq!(value["splits"]["test"]["unknown_errors"], 0);
    }
    let mini = report.endpoints[1].training.as_ref().unwrap();
    assert_eq!(mini.steps[1].start, 64);
    assert_eq!(mini.steps[1].count, 64);
    assert_eq!(mini.steps[2].start, 61);
    assert_eq!(
        report.endpoints[2].training.as_ref().unwrap().model,
        report.initialization.as_ref().unwrap().model
    );
    let mut changed = report.clone();
    changed.endpoints[0]
        .training
        .as_mut()
        .unwrap()
        .model
        .coefficients[0] += 0.01;
    assert!(binding::check(&stage.request, &changed).is_err());
    let mut changed = report.clone();
    changed.endpoints[2]
        .training
        .as_mut()
        .unwrap()
        .work
        .gradient_batches = 1;
    assert!(binding::check(&stage.request, &changed).is_err());
    Ok(())
}

#[test]
fn inconsistent_training_retains_all_three_refusals() -> Result<()> {
    let mut data = fixture();
    data.task.splits.get_mut("train").unwrap()[2].targets[2] = 2;
    let stage = runner::construct(4, &data.task.splits["train"], data.metadata())?;
    let report = stage.model.as_ref().unwrap();
    assert!(report.discovery.reason.is_some());
    assert!(report.initialization_error.is_some());
    binding::replay(&stage.request, report)?;
    let scored = runner::measurements(&stage, &data)?;
    assert_eq!(scored["modes"].as_array().unwrap().len(), 3);
    for mode in scored["modes"].as_array().unwrap() {
        assert!(mode["error"].is_string());
        assert!(mode["measurement"].is_null());
        assert!(mode["primary"].is_null());
        assert_eq!(mode["all_heldout99"], false);
    }
    Ok(())
}

#[test]
fn raw_and_exact_goals_and_fresh_artifacts_cannot_be_conflated() -> Result<()> {
    let value = goals(&json!({"token_accuracy":0.995,"sequence_accuracy":0.96}));
    assert_eq!(
        value,
        json!({"raw99":true,"exact99":false,"both99":false,"original_success":true})
    );
    let dir = std::env::temp_dir().join(format!("sequences-fixture-{}", std::process::id()));
    std::fs::create_dir(&dir)?;
    let path = dir.join("state.json");
    runner::artifacts::write(&path, &value)?;
    assert!(runner::artifacts::write(&path, &json!({"replacement":true})).is_err());
    assert_eq!(runner::artifacts::read::<Value>(&path)?, value);
    std::fs::remove_dir_all(dir)?;
    Ok(())
}

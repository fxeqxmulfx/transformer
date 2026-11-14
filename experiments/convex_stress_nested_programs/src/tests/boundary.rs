use super::*;
use serde_json::json;
fn row(tokens: &[usize], label: usize) -> data::Example {
    let mut targets = vec![-100; tokens.len()];
    *targets.last_mut().unwrap() = label as i64;
    data::Example {
        task: "fixture".into(),
        tokens: tokens.iter().map(|&t| t as i64).collect(),
        changes: targets.iter().map(|&t| t >= 0).collect(),
        targets,
        prompt: vec![],
        answer: vec![],
        limit: 0,
        field: None,
        row_index: None,
    }
}
fn dataset() -> dataset::Dataset {
    let input = fixture();
    let test = vec![row(&[0, 1, 3, 4, 2], 6), row(&[0, 1, 3, 2, 4], 8)];
    dataset::Dataset {
        name: "fixture".into(),
        task: data::Task {
            vocab: 10,
            selection: "validation".into(),
            splits: [
                (
                    "train".into(),
                    input
                        .observations
                        .iter()
                        .map(|s| row(&s.tokens, s.label))
                        .collect(),
                ),
                ("validation".into(), test.clone()),
                ("test".into(), test),
                ("test_long".into(), vec![row(&deep(), 6)]),
            ]
            .into(),
        },
        original_fingerprints: Default::default(),
        tensor_fingerprints: Default::default(),
        empty_novel_splits: vec![],
    }
}
#[test]
fn training_boundary_excludes_names_and_all_heldout_labels() -> Result<()> {
    let mut data = dataset();
    let original = runner::construct(10, &data.task.splits["train"], json!({"name":"original"}))?;
    assert!(original.model.as_ref().unwrap().latent.is_some());
    for (split, rows) in &mut data.task.splits {
        for row in rows {
            row.task = "arbitrary-identity".into();
            row.changes.fill(false);
            if split != "train" {
                row.tokens.fill(7);
                row.targets.fill(7);
            }
        }
    }
    let changed = runner::construct(10, &data.task.splits["train"], json!({"name":"changed"}))?;
    assert_eq!(original.input, changed.input);
    assert_eq!(original.model, changed.model);
    Ok(())
}
#[test]
fn every_mode_scores_all_prefixes_and_preserves_missing_state_diagnostics() -> Result<()> {
    let data = dataset();
    let stage = runner::construct(10, &data.task.splits["train"], data.metadata())?;
    let result = runner::measurements(&stage, &data)?;
    assert_eq!(result["modes"].as_array().unwrap().len(), 3);
    for mode in result["modes"].as_array().unwrap() {
        assert!(mode["error"].is_null());
        assert_eq!(mode["all_heldout99"], true);
        assert_eq!(
            mode["measurement"]["splits"]["test"]["predictions"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
    }
    let round: runner::Stage = serde_json::from_slice(&serde_json::to_vec(&stage)?)?;
    binding::check(&round.input, round.model.as_ref().unwrap())?;
    assert!(runner::artifacts::same_json(
        &result,
        &runner::measurements(&round, &data)?
    )?);
    let input = Input {
        vocabulary: 3,
        context: 2,
        observations: vec![Observation {
            tokens: vec![0, 1],
            label: 2,
        }],
    };
    let model = model::Model::new(&discovery::run(input.clone())?)?;
    let evaluated = evaluation::evaluate(
        &model,
        &input,
        &[("test".into(), vec![row(&[0, 0, 2], 1)])].into(),
    )?;
    assert_eq!(evaluated["splits"]["test"]["any_missing_targets"], 1);
    assert_eq!(evaluated["splits"]["test"]["ambiguous_targets"], 1);
    assert_eq!(
        evaluated["splits"]["test"]["unseen_input_symbol_targets"],
        1
    );
    assert_eq!(evaluated["splits"]["test"]["unseen_label_targets"], 1);
    assert_eq!(
        evaluated["splits"]["test"]["predictions"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    Ok(())
}
#[test]
fn parent_priority_preserves_reviewed_models_and_unseen_symbol_predictions() -> Result<()> {
    let rows = (4..=6)
        .flat_map(|a| {
            (4..=6).filter(move |&b| a != b).flat_map(move |b| {
                [
                    Observation {
                        tokens: vec![1, 2, a, 3, b, 2],
                        label: a,
                    },
                    Observation {
                        tokens: vec![1, 2, a, 3, b, 3],
                        label: b,
                    },
                ]
            })
        })
        .collect();
    let input = Input {
        vocabulary: 8,
        context: 6,
        observations: rows,
    };
    let expected = parent::binding::run(input.clone())?;
    let actual = binding::run(input.clone())?;
    assert_eq!(actual.parent.as_ref(), Some(&expected));
    assert!(actual.latent.is_none() && actual.latent_error.is_none());
    binding::check(&input, &actual)?;
    for mode in &actual.parent.as_ref().unwrap().copy.endpoints {
        let model = &mode.training.as_ref().unwrap().model;
        assert_eq!(argmax(&model.logits_tokens(&[1, 2, 7, 3, 6, 2])), 7);
    }
    Ok(())
}
#[test]
fn all_refusals_and_original_success_remain_in_denominators() -> Result<()> {
    let mut data = dataset();
    let mut collision = data.task.splits["train"][0].clone();
    *collision.targets.last_mut().unwrap() = 9;
    data.task.splits.get_mut("train").unwrap().push(collision);
    let stage = runner::construct(10, &data.task.splits["train"], data.metadata())?;
    let measured = runner::measurements(&stage, &data)?;
    assert_eq!(measured["modes"].as_array().unwrap().len(), 3);
    for mode in measured["modes"].as_array().unwrap() {
        assert!(mode["error"].is_string());
        assert!(mode["measurement"].is_null());
        assert_eq!(mode["all_heldout99"], false);
    }
    assert_eq!(
        sequences::adapter::goals(&json!({"token_accuracy":0.995,"sequence_accuracy":0.96})),
        json!({"raw99":true,"exact99":false,"both99":false,"original_success":true})
    );
    Ok(())
}

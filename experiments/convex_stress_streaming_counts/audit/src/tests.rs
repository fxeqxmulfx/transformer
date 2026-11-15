use crate::*;
use convex_stress_streaming_counts::{discovery, prefix};
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, path::PathBuf};

fn temporary(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("stream347-readback-{}-{name}", std::process::id()))
}
fn forward() -> joint::training::Policy {
    joint::training::Policy {
        mode: joint::training::Mode::Forward,
        epochs: 0,
        batch: 2,
        learning_rate: 1e5,
        initial_total_mass: 131072.,
        projection_cycles: 20000,
        projection_tolerance: 1e-10,
    }
}
fn example(prompt: Vec<i64>, answer: Vec<i64>, limit: usize) -> dataset::data::Example {
    let tokens = prompt
        .iter()
        .chain(&answer[..answer.len() - 1])
        .copied()
        .collect();
    let targets: Vec<_> = std::iter::repeat_n(-100, prompt.len() - 1)
        .chain(answer.iter().copied())
        .collect();
    let mut previous = None;
    let changes = targets
        .iter()
        .map(|&y| {
            let c = y >= 0 && previous != Some(y);
            if y >= 0 {
                previous = Some(y);
            }
            c
        })
        .collect();
    dataset::data::Example {
        tokens,
        targets,
        changes,
        prompt,
        answer,
        limit,
        field: None,
        row_index: None,
        task: "hand_audit".into(),
    }
}

#[test]
fn complete_rollouts_recount_correct_unseen_answers_early_eos_and_extra_tokens() -> Result<()> {
    let train = vec![
        example(vec![1, 100, 102, 18], vec![100, 101, 102, 17], 6),
        example(vec![1, 102, 104, 18], vec![102, 103, 104, 17], 6),
    ];
    let input = adapter::input(229, &train)?;
    let d = discovery::run(
        &input.observations,
        observations::coordinates(229)?,
        vec![program::Program::Range {
            reverse: false,
            inclusive: true,
        }],
    )?;
    let splits = BTreeMap::from([
        ("train".into(), train),
        (
            "good".into(),
            vec![example(
                vec![1, 103, 106, 18],
                vec![103, 104, 105, 106, 17],
                7,
            )],
        ),
        (
            "early".into(),
            vec![example(vec![1, 103, 103, 18], vec![103, 104, 105, 17], 7)],
        ),
        (
            "overlong".into(),
            vec![example(vec![1, 103, 106, 18], vec![103, 17], 7)],
        ),
    ]);
    for prior in [model::Prior::Coordinates, model::Prior::Categorical] {
        let t = model::train(&d, &input.observations, prior, forward(), 1e9, 1.)?;
        let path = temporary(&format!("rollout-{prior:?}.jsonl"));
        let saved = evaluation::evaluate(&t.model, &input, &splits, &path)?;
        let audited = raw::verify(&t.model, &input, &splits, &path, &saved)?;
        assert_eq!(audited["all_source_rows_recounted"], true);
        assert_eq!(
            audited["original_training_references"],
            input.observations.len()
        );
        assert_eq!(saved["splits"]["early"]["metrics"]["exact"], 0);
        assert_eq!(saved["splits"]["early"]["metrics"]["eos_rate"], 1.);
        assert!(
            saved["splits"]["overlong"]["metrics"]["extra_tokens"]
                .as_u64()
                .unwrap()
                > 0
        );
        if prior == model::Prior::Coordinates {
            assert_eq!(saved["splits"]["good"]["metrics"]["exact"], 1);
            assert_eq!(saved["splits"]["good"]["unseen_label_correct"], 2);
        } else {
            assert_eq!(saved["splits"]["good"]["metrics"]["exact"], 0);
        }
        let mut forged = saved.clone();
        forged["splits"]["early"]["metrics"]["exact"] = json!(1);
        assert!(raw::verify(&t.model, &input, &splits, &path, &forged).is_err());
        fs::remove_file(path)?;
    }
    Ok(())
}

#[test]
fn prefix_recount_rejects_changed_ce_counts_missing_splits_and_rehashed_tampering() -> Result<()> {
    let e = dataset::data::Example {
        tokens: vec![1, 9, 10, 10, 23, 9],
        targets: vec![-100, 15, 15, 16, 16, 15],
        changes: vec![false, true, false, true, false, true],
        prompt: vec![],
        answer: vec![],
        limit: 0,
        field: None,
        row_index: None,
        task: "hand_prefix".into(),
    };
    let input = adapter::input(36, std::slice::from_ref(&e))?;
    let f = prefix::Formula::Less(
        Box::new(prefix::Term::Count(Box::new(prefix::Formula::Symbol(9)))),
        Box::new(prefix::Term::Count(Box::new(prefix::Formula::Symbol(10)))),
    );
    let d = discovery::run(
        &input.observations,
        observations::coordinates(36)?,
        vec![program::Program::Prefix(f)],
    )?;
    let t = model::train(
        &d,
        &input.observations,
        model::Prior::Coordinates,
        forward(),
        1e9,
        1.,
    )?;
    let splits = BTreeMap::from([("train".into(), vec![e.clone()]), ("test".into(), vec![e])]);
    let path = temporary("prefix.jsonl");
    let saved = evaluation::evaluate(&t.model, &input, &splits, &path)?;
    let verified = raw::verify(&t.model, &input, &splits, &path, &saved)?;
    assert_eq!(verified["original_query_positions"], 10);
    assert_eq!(saved["splits"]["test"]["metrics"]["transition_targets"], 3);
    let mut forged = saved.clone();
    forged["splits"]["test"]["metrics"]["teacher_forced"]["loss"] = json!(1.);
    assert!(raw::verify(&t.model, &input, &splits, &path, &forged).is_err());
    forged = saved.clone();
    forged["splits"].as_object_mut().unwrap().remove("train");
    assert!(raw::verify(&t.model, &input, &splits, &path, &forged).is_err());
    let mut records: Vec<Value> = fs::read_to_string(&path)?
        .lines()
        .map(serde_json::from_str)
        .collect::<std::result::Result<_, _>>()?;
    records[0]["ce"] = json!(0.25);
    let changed = records.iter().map(|r| format!("{r}\n")).collect::<String>();
    fs::write(&path, changed)?;
    forged = saved.clone();
    forged["raw_sha256"] = json!(artifacts::digest(&path)?);
    assert!(raw::verify(&t.model, &input, &splits, &path, &forged).is_err());
    fs::remove_file(path)?;
    Ok(())
}

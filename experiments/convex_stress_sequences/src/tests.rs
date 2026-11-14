use crate::*;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

#[test]
fn shared_generators_match_frozen_original35_fingerprints() -> Result<()> {
    let suite: Value = serde_json::from_str(include_str!("../../convex_stress/suite.json"))?;
    for (name, spec) in [
        (
            "gpt-probe-key-value-binding",
            json!({"task":"mqar","length":12,"min_length":null,"symbols":32,"pairs":2,"queries":1,"query_gap":0,"alpha":0.1}),
        ),
        (
            "basis-depth-easy",
            json!({"task":"blocks","length":64,"min_length":32,"neutral_fraction":0.25,"max_neutral_gap":null,"blocks":2}),
        ),
        (
            "basis-depth-hard",
            json!({"task":"blocks","length":64,"min_length":32,"neutral_fraction":0.25,"max_neutral_gap":null,"blocks":4}),
        ),
    ] {
        let row = suite["tasks"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["name"] == name)
            .unwrap();
        for split in ["train", "validation", "test"] {
            let count = if split == "train" {
                row["train_examples"].as_u64().unwrap() as usize
            } else {
                512
            };
            let rows = dataset::generate(&spec, split, count)?;
            assert_eq!(
                row["data_fingerprints"][split],
                wire::split_hash(&spec, split, &rows),
                "{name}/{split}"
            );
        }
        if name == "basis-depth-hard" {
            let mut longer = spec.clone();
            longer["length"] = json!(128);
            longer["min_length"] = Value::Null;
            let rows = dataset::generate(&longer, "validation", 512)?;
            assert_eq!(
                row["data_fingerprints"]["validation/length-128"],
                wire::split_hash(&longer, "validation", &rows)
            );
        }
    }
    Ok(())
}

fn independently_expected(e: &data::Example) -> Vec<i64> {
    let mut expected = vec![-100; e.tokens.len()];
    match e.task.as_str() {
        "mqar" => {
            let table: BTreeMap<_, _> = e.tokens[1..9]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|r| (r[0], r[1]))
                .collect();
            for (i, &token) in e.tokens.iter().enumerate().skip(9) {
                if let Some(&value) = table.get(&token) {
                    expected[i] = value;
                }
            }
        }
        "lookup" => {
            let table: BTreeMap<_, _> = e.tokens[1..17]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|r| {
                    assert_eq!((r[0], r[2]), (2, 3));
                    (r[1], r[3])
                })
                .collect();
            assert_eq!(table.len(), 4);
            assert_eq!(e.tokens[17], 5);
            for i in 18..e.tokens.len() - 1 {
                if e.tokens[i] == 4 {
                    let k = e.tokens[i + 1];
                    expected[i + 1] = table[&table[&k]];
                }
            }
        }
        "dyck" => {
            for (i, value) in expected.iter_mut().enumerate().skip(1) {
                let balances: Vec<_> = (1..=i)
                    .map(|end| {
                        let part = &e.tokens[1..=end];
                        part.iter().filter(|&&t| t == 7).count() as i64
                            - part.iter().filter(|&&t| t == 8).count() as i64
                    })
                    .collect();
                *value = if balances.iter().any(|&b| b < 0) {
                    14
                } else if balances.last() == Some(&0) {
                    12
                } else {
                    13
                };
            }
        }
        "blocks" => {
            for (i, value) in expected.iter_mut().enumerate().skip(1) {
                let active: Vec<_> = e.tokens[1..=i]
                    .iter()
                    .copied()
                    .filter(|&t| t != 11)
                    .collect();
                let runs = usize::from(!active.is_empty())
                    + active.windows(2).filter(|w| w[0] != w[1]).count();
                *value = if active.first() == Some(&9) && runs == 3 {
                    16
                } else {
                    15
                };
            }
        }
        _ => panic!("test oracle has no such family"),
    }
    expected
}

#[test]
fn all_first_cohort_splits_match_independent_oracles_masks_and_novel_inputs() -> Result<()> {
    let catalog: Value = serde_json::from_str(CATALOG)?;
    for name in catalog["first_cohort"].as_array().unwrap() {
        let data = dataset::build(name.as_str().unwrap())?;
        let train: BTreeSet<_> = data.task.splits["train"]
            .iter()
            .map(|e| e.tokens.clone())
            .collect();
        assert_eq!(data.task.splits["train"].len(), 128);
        assert_eq!(data.task.splits["validation"].len(), 32);
        assert_eq!(data.task.splits["test"].len(), 64);
        assert_eq!(data.original_fingerprints.len(), 8);
        assert_eq!(
            data.tensor_fingerprints["train"],
            data.tensor_fingerprints["train_clean"]
        );
        assert!(data.task.splits.len() >= 8 && data.task.splits.len() <= 11);
        for (split, rows) in &data.task.splits {
            assert_eq!(data.tensor_fingerprints[split], wire::tensor_hash(rows));
            for e in rows {
                assert_eq!(e.targets, independently_expected(e));
                assert!(e.prompt.is_empty() && e.answer.is_empty() && e.limit == 0);
                assert!(
                    e.tokens
                        .iter()
                        .all(|&t| t >= 0 && (t as usize) < data.task.vocab)
                );
                assert!(
                    e.targets
                        .iter()
                        .all(|&t| t == -100 || (t >= 0 && (t as usize) < data.task.vocab))
                );
                if split.ends_with("/novel") {
                    assert!(!train.contains(&e.tokens));
                }
                let mut previous = None;
                for (&y, &changed) in e.targets.iter().zip(&e.changes) {
                    assert_eq!(changed, y >= 0 && previous != Some(y));
                    if y >= 0 {
                        previous = Some(y);
                    }
                }
            }
        }
        if name == "suite-lookup" {
            for pair in data.task.splits["train"].as_chunks::<2>().0 {
                assert_ne!(pair[0].targets, pair[1].targets);
                let mut first = pair[0].tokens.clone();
                let mut second = pair[1].tokens.clone();
                first.sort_unstable();
                second.sort_unstable();
                assert_eq!(first, second);
            }
        }
        assert_eq!(
            data.metadata(),
            dataset::build(name.as_str().unwrap())?.metadata()
        );
    }
    Ok(())
}

#[test]
fn invalid_generator_dimensions_and_unfrozen_cases_refuse() -> Result<()> {
    let spec = json!({"task":"mqar","length":24,"min_length":24,"symbols":64,"pairs":4,"queries":2,"query_gap":0,"alpha":0.1});
    for (field, value) in [
        ("length", json!(1)),
        ("pairs", json!(0)),
        ("queries", json!(9)),
        ("overwrites", json!(1)),
        ("alpha", json!(-1)),
    ] {
        let mut bad = spec.clone();
        bad[field] = value;
        assert!(dataset::generate(&bad, "train", 128).is_err());
    }
    assert!(dataset::generate(&spec, "invalid", 128).is_err());
    assert!(dataset::generate(&spec, "train", 0).is_err());
    assert!(dataset::build("suite-copy-repeat").is_err());
    Ok(())
}

#[test]
fn raw99_does_not_replace_exact_sequence99() -> Result<()> {
    struct OneError;
    impl metrics::Predictor for OneError {
        fn vocab(&self) -> usize {
            3
        }
        fn logits(&self, prefix: &[i64]) -> Vec<f64> {
            if prefix.last() == Some(&2) {
                vec![24., 0., 0.]
            } else {
                vec![0., 24., 0.]
            }
        }
    }
    let first = wire::example("metric-fixture", vec![1; 100], vec![1; 100]);
    let mut second = first.clone();
    second.tokens[99] = 2;
    let report = metrics::evaluate(&OneError, &[first, second])?;
    assert_eq!(report["correct"], 199);
    assert_eq!(report["target_count"], 200);
    assert_eq!(report["token_accuracy"], 0.995);
    assert_eq!(report["sequence_accuracy"], 0.5);
    assert_eq!(report["exact"], 1);
    assert_eq!(report["example_count"], 2);
    Ok(())
}

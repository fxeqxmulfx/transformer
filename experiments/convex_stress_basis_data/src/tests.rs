use crate::*;
use serde_json::{Value, json};
use std::collections::BTreeSet;

#[test]
fn complete_original_pools_masks_vocabulary_and_selection() -> Result<()> {
    for case in CASES {
        let (d, noise) = dataset::build(case)?;
        let parity = case == "basis-parity-noise20";
        let language = case.starts_with("language-");
        assert_eq!(
            d.task.splits["train"].len(),
            if parity { 512 } else { 20000 }
        );
        assert_eq!(
            d.task.splits["validation"].len(),
            if parity { 128 } else { 512 }
        );
        assert_eq!(d.task.splits["test"].len(), if parity { 256 } else { 512 });
        assert_eq!(
            d.task.vocab,
            if parity {
                68
            } else if language {
                36
            } else {
                548
            }
        );
        let primary = if language {
            "validation/length-128/novel"
        } else {
            "validation/novel"
        };
        assert_eq!(d.task.selection, primary);
        assert!(d.empty_novel_splits.is_empty());
        assert_eq!(
            d.task.splits.len(),
            if language {
                6
            } else if parity {
                5
            } else {
                4
            }
        );
        for (name, rows) in &d.task.splits {
            assert!(rows.iter().all(|e| {
                e.tokens
                    .iter()
                    .all(|&t| t >= 0 && (t as usize) < d.task.vocab)
            }));
            assert!(rows.iter().all(|e| {
                e.targets
                    .iter()
                    .all(|&t| t == -100 || (t >= 0 && (t as usize) < d.task.vocab))
            }));
            if name.contains("length-128") {
                assert!(rows.iter().all(|e| e.tokens.len() == 128));
            }
        }
        if parity {
            assert_eq!(noise["eligible_targets"], 512);
            assert!(noise["changed_targets"].as_u64().unwrap() > 0);
        } else {
            assert!(noise.is_null());
            assert!(!d.task.splits.contains_key("train_clean"));
        }
    }
    Ok(())
}

#[test]
fn original_fingerprints_match_preexisting_native_ports() -> Result<()> {
    // Immutable original35 input fingerprints, never regenerated from this port.
    let suite: Value = serde_json::from_str(include_str!("../../convex_stress/suite.json"))?;
    for case in [
        "basis-depth-easy",
        "basis-depth-hard",
        "basis-recall-easy",
        "basis-parity16",
    ] {
        let row = suite["tasks"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["name"] == case)
            .unwrap();
        let mut spec = if case.contains("depth") {
            let mut s = dataset::specification(if case.ends_with("easy") {
                "language-l2"
            } else {
                "language-l4"
            })?;
            s["neutral_fraction"] = json!(0.25);
            s
        } else if case.contains("recall") {
            let mut s = dataset::specification("basis-recall-hard")?;
            s["pairs"] = json!(8);
            s["min_length"] = json!(32);
            s.as_object_mut().unwrap().remove("overwrites");
            s
        } else {
            dataset::specification("basis-parity-noise20")?
        };
        let rows = generator::generate(&spec, "test", 512, &BTreeSet::new(), false)?;
        assert_eq!(
            wire::split_hash(&spec, "test", &rows),
            row["data_fingerprints"]["test"].as_str().unwrap(),
            "{case}"
        );
        if case.contains("depth") {
            spec["length"] = json!(128);
            spec["min_length"] = Value::Null;
            let rows = generator::generate(&spec, "test", 512, &BTreeSet::new(), false)?;
            assert_eq!(
                wire::split_hash(&spec, "test", &rows),
                row["data_fingerprints"]["test/length-128"]
                    .as_str()
                    .unwrap()
            );
        }
    }
    Ok(())
}

#[test]
fn latest_write_and_masks_reject_earlier_values() -> Result<()> {
    let s = json!({"task":"mqar","symbols":2,"pairs":4,"queries":2,"query_gap":0,"alpha":0.1,"overwrites":2,"length":11,"min_length":null});
    let e = wire::example(
        "mqar",
        vec![1, 36, 38, 37, 39, 36, 39, 37, 38, 36, 37],
        vec![-100, -100, -100, -100, -100, -100, -100, -100, -100, 39, 38],
    );
    independent::check(&s, &e)?;
    let mut wrong = e.clone();
    wrong.targets[9] = 38;
    assert!(independent::check(&s, &wrong).is_err());
    let mut same = e.clone();
    same.tokens[6] = 38;
    assert!(independent::check(&s, &same).is_err());
    let mut mask = e.clone();
    mask.changes[9] = false;
    assert!(independent::check(&s, &mask).is_err());
    let rows = generator::generate(&s, "train", 31, &BTreeSet::new(), false)?;
    assert_eq!(rows.len(), 31);
    Ok(())
}

#[test]
fn neutral_free_single_block_and_multiple_block_labels() -> Result<()> {
    let mut s = dataset::specification("language-l1")?;
    s["length"] = json!(5);
    s["min_length"] = Value::Null;
    independent::check(
        &s,
        &wire::example("blocks", vec![1, 9, 9, 10, 9], vec![-100, 16, 16, 15, 15]),
    )?;
    let bad = wire::example("blocks", vec![1, 9, 11, 9, 9], vec![-100, 16, 16, 16, 16]);
    assert!(independent::check(&s, &bad).is_err());
    s["blocks"] = json!(3);
    independent::check(
        &s,
        &wire::example("blocks", vec![1, 9, 10, 9, 9], vec![-100, 15, 15, 16, 16]),
    )?;
    for blocks in 1..=6 {
        s["blocks"] = json!(blocks);
        s["length"] = json!(16);
        let rows = generator::generate(&s, "validation", 33, &BTreeSet::new(), false)?;
        assert!(rows.iter().all(|e| !e.tokens.contains(&11)));
    }
    Ok(())
}

#[test]
fn disjoint_pools_and_noise_are_fixed_on_original_prompt() -> Result<()> {
    let (d, report) = dataset::build("basis-parity-noise20")?;
    let mut seen = BTreeSet::new();
    for name in ["train_clean", "validation", "test"] {
        for row in &d.task.splits[name] {
            assert!(seen.insert(generator::key(row).to_vec()));
        }
    }
    let (again, repeated) = noise::apply(&d.task.splits["train_clean"], 0.2, 2)?;
    assert_eq!(report, repeated);
    assert_eq!(wire::tensor_hash(&again), d.tensor_fingerprints["train"]);
    let mut actual = 0;
    for (clean, observed) in d.task.splits["train_clean"].iter().zip(&again) {
        assert_eq!(clean.prompt, observed.prompt);
        assert_eq!(observed.answer[1], 17);
        assert_eq!(observed.tokens.last(), Some(&observed.answer[0]));
        actual += usize::from(clean.answer[0] != observed.answer[0]);
        let (zero, flip) = noise::corrupt(clean, 0., 2)?;
        assert!(!flip);
        assert_eq!(wire::value(&zero), wire::value(clean));
        let (all, flip) = noise::corrupt(clean, 1., 2)?;
        assert!(flip);
        assert_eq!(all.answer[0], 49 - clean.answer[0]);
    }
    assert_eq!(report["changed_targets"], actual);
    assert!(noise::apply(&[], 0.2, 2).is_err());
    Ok(())
}

#[test]
fn tiny_stream_caps_and_independent_rollout_semantics() -> Result<()> {
    let mut s = dataset::specification("basis-parity-noise20")?;
    s["length"] = json!(1);
    s["min_length"] = json!(1);
    assert!(generator::generate(&s, "train", 3, &BTreeSet::new(), true).is_err());
    assert!(generator::generate(&s, "unknown", 1, &BTreeSet::new(), false).is_err());
    assert!(generator::generate(&s, "train", 20001, &BTreeSet::new(), false).is_err());
    let e = wire::answered("parity", vec![1, 22, 18], vec![25, 17], 2);
    let mut visited = vec![];
    let output = generative::scoring::rollout(&e.prompt, e.limit, 68, |prefix| {
        visited.push(prefix.to_vec());
        Ok(if prefix.len() == 3 { 24 } else { 17 })
    })?;
    assert_eq!(visited[1], vec![1, 22, 18, 24]);
    let score = generative::scoring::score(&e.answer, &output)?;
    assert_eq!((score.targets, score.correct, score.exact), (2, 1, false));
    Ok(())
}

#[test]
fn original_recall_denominators_and_independent_noise_addresses() -> Result<()> {
    use sha2::{Digest, Sha256};
    let spec = dataset::specification("basis-recall-hard")?;
    let rows = generator::generate(&spec, "test", 512, &BTreeSet::new(), false)?;
    // Frozen f31ed0f5 history: one available final_test; other7 cases pending.
    assert_eq!(
        rows.iter()
            .flat_map(|e| &e.targets)
            .filter(|&&y| y >= 0)
            .count(),
        4096
    );
    assert_eq!(
        rows.iter().flat_map(|e| &e.changes).filter(|&&b| b).count(),
        4082
    );
    let mut flips = 0;
    for mask in 0..64u32 {
        let bits: Vec<_> = (0..6)
            .map(|i| 21 + i64::from(mask & (1 << i) != 0))
            .collect();
        let mut prompt = vec![1];
        prompt.extend(bits);
        prompt.push(18);
        let clean = wire::answered(
            "parity",
            prompt.clone(),
            vec![24 + i64::from(mask.count_ones() % 2 != 0), 17],
            2,
        );
        // Build the literal source JSON address separately from wire::json_text.
        let integers = prompt
            .iter()
            .map(i64::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        let text = format!("[2, \"parity\", [[{integers}], 0], [24, 25]]");
        let digest = Sha256::digest(text.as_bytes());
        let mut words: Vec<_> = digest
            .as_chunks::<4>()
            .0
            .iter()
            .rev()
            .map(|b| u32::from_be_bytes(*b))
            .collect();
        while words.len() > 1 && words.last() == Some(&0) {
            words.pop();
        }
        let expected = Random::words(&words).random() < 0.2;
        let (observed, flip) = noise::corrupt(&clean, 0.2, 2)?;
        assert_eq!(flip, expected);
        assert_eq!(
            observed.answer[0],
            if expected {
                49 - clean.answer[0]
            } else {
                clean.answer[0]
            }
        );
        flips += usize::from(flip);
    }
    assert!(flips > 0 && flips < 64);
    Ok(())
}

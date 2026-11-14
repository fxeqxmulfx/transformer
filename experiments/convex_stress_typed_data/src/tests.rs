use crate::*;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

/// Independent oracle: repeatedly cancel adjacent matched pairs. Any closing
/// symbol left in the reduced word makes a prefix invalid; no stack is used.
fn reduced_status(tokens: &[i64], types: usize) -> i64 {
    let pairs = generator::pairs(types);
    let mut word: Vec<_> = tokens
        .iter()
        .copied()
        .filter(|&t| t != 1 && t != 11)
        .collect();
    while let Some(i) = word.windows(2).position(|w| pairs.contains(&(w[0], w[1]))) {
        word.drain(i..i + 2);
    }
    if word.iter().any(|t| pairs.iter().any(|(_, c)| c == t)) {
        14
    } else if word.is_empty() {
        12
    } else {
        13
    }
}
#[test]
fn original_splits_masks_vocab_and_independent_reduction_oracle() -> Result<()> {
    for (case, types, test_targets) in [(CASES[0], 2, 842), (CASES[1], 3, 850)] {
        let dataset = dataset::build(case)?;
        assert_eq!(dataset.task.vocab, 36 + 2 * (types - 1));
        assert_eq!(dataset.task.splits.len(), 11);
        assert_eq!(dataset.task.splits["train"].len(), 128);
        assert_eq!(dataset.task.splits["validation"].len(), 32);
        assert_eq!(dataset.task.splits["test"].len(), 64);
        // Frozen historical endpoint target counts; this is not a claim of
        // unavailable historical per-example hash identity.
        assert_eq!(
            dataset.task.splits["test"]
                .iter()
                .map(|r| r.tokens.len() - 1)
                .sum::<usize>(),
            test_targets
        );
        assert_eq!(dataset.task.selection, "validation/novel");
        assert_eq!(
            dataset.tensor_fingerprints["train"],
            dataset.tensor_fingerprints["train_clean"]
        );
        let train: BTreeSet<_> = dataset.task.splits["train"]
            .iter()
            .map(|r| &r.tokens)
            .collect();
        for (name, rows) in &dataset.task.splits {
            for row in rows {
                assert_eq!(row.task, "dyck2");
                assert!(row.prompt.is_empty() && row.answer.is_empty() && row.limit == 0);
                assert_eq!(row.tokens[0], 1);
                assert_eq!(row.targets[0], -100);
                assert!(!row.changes[0]);
                assert_eq!(row.tokens.len(), row.targets.len());
                assert!(
                    row.tokens
                        .iter()
                        .all(|&t| t >= 0 && t < dataset.task.vocab as i64)
                );
                if name.ends_with("/novel") {
                    assert!(!train.contains(&row.tokens));
                }
                for i in 1..row.tokens.len() {
                    assert_eq!(row.targets[i], reduced_status(&row.tokens[..=i], types));
                    assert_eq!(
                        row.changes[i],
                        i == 1 || row.targets[i] != row.targets[i - 1]
                    );
                }
            }
        }
        for length in [32, 64] {
            let rows = &dataset.task.splits[&format!("test/length-{length}")];
            assert_eq!(
                rows.iter().map(|r| r.tokens.len() - 1).sum::<usize>(),
                64 * (length - 1)
            );
        }
    }
    Ok(())
}
#[test]
fn matched_words_preserve_counts_and_introduce_type_mismatch() {
    for types in [2, 3] {
        let spec = json!({"test":"paired-conformance","types":types});
        let mut rng = wire::rng(&spec, "train");
        for size in (4..=24).step_by(2) {
            for _ in 0..16 {
                let (a, b) = generator::matched(types, size, 8, &mut rng);
                let counts = |word: &[i64]| {
                    let mut out = BTreeMap::new();
                    for &t in word {
                        *out.entry(t).or_insert(0) += 1;
                    }
                    out
                };
                assert_eq!(counts(&a), counts(&b));
                assert_ne!(a, b);
                assert_eq!(reduced_status(&a, types), 12);
                assert_eq!(reduced_status(&b, types), 14);
            }
        }
    }
}
#[test]
fn same_net_balance_with_wrong_type_stays_invalid_after_later_closes() {
    let good = generator::labels(vec![1, 7, 36, 37, 8, 11], 2);
    let bad = generator::labels(vec![1, 7, 36, 8, 37, 11], 2);
    assert_eq!(good.targets, vec![-100, 13, 13, 13, 12, 12]);
    assert_eq!(bad.targets, vec![-100, 13, 13, 14, 14, 14]);
}
#[test]
fn deterministic_stream_prefixes_and_bounds_are_preserved() -> Result<()> {
    let p: Value = serde_json::from_str(PROTOCOL)?;
    let mut spec = p["definitions"][CASES[0]].clone();
    spec.as_object_mut().unwrap().remove("longer_lengths");
    let a = dataset::generate(&spec, "train", 8)?;
    let b = dataset::generate(&spec, "train", 13)?;
    assert_eq!(wire::tensor_hash(&a), wire::tensor_hash(&b[..8]));
    assert_ne!(
        wire::tensor_hash(&a),
        wire::tensor_hash(&dataset::generate(&spec, "test", 8)?)
    );
    assert!(dataset::generate(&spec, "unknown", 8).is_err());
    assert!(dataset::generate(&spec, "train", 0).is_err());
    spec["bracket_types"] = json!(1);
    assert!(dataset::generate(&spec, "train", 8).is_err());
    assert!(dataset::build("suite-dyck").is_err());
    Ok(())
}
#[test]
fn changed_frozen_bytes_and_overwritten_outputs_refuse() -> Result<()> {
    let dir = std::env::temp_dir().join(format!("typed-data-input-check-{}", std::process::id()));
    std::fs::create_dir(&dir)?;
    let file = dir.join("input.json");
    let bytes = b"{\"source\":1}\n";
    std::fs::write(&file, bytes)?;
    assert_eq!(
        inventory::read_pinned(&file, &hash(bytes))?,
        json!({"source":1})
    );
    std::fs::write(&file, b"{\"source\":2}\n")?;
    assert!(inventory::read_pinned(&file, &hash(bytes)).is_err());
    assert!(inventory::prepare(&file, &file, &file).is_err());
    std::fs::remove_dir_all(dir)?;
    Ok(())
}

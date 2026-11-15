use crate::*;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

fn ranked(word: &[i64]) -> Vec<i64> {
    let mut counts = BTreeMap::new();
    for &token in word {
        *counts.entry(token).or_insert(0usize) += 1;
    }
    counts
        .into_iter()
        .flat_map(|(token, count)| std::iter::repeat_n(token, count))
        .collect()
}

#[test]
fn original_denominators_causal_masks_and_independent_word_oracles() -> Result<()> {
    // Target counts come from the pinned historical GPT snapshot, not this port.
    for (case, targets) in CASES.into_iter().zip([772, 842, 842, 842, 828, 860]) {
        let d = dataset::build(case)?;
        assert_eq!(d.task.vocab, 100);
        assert_eq!(d.task.splits.len(), 11);
        assert_eq!(d.task.splits["train"].len(), 128);
        assert_eq!(d.task.splits["validation"].len(), 32);
        assert_eq!(d.task.splits["test"].len(), 64);
        assert_eq!(
            d.task.splits["test"]
                .iter()
                .map(|r| r.answer.len())
                .sum::<usize>(),
            targets
        );
        assert_eq!(
            d.tensor_fingerprints["train"],
            d.tensor_fingerprints["train_clean"]
        );
        let train: BTreeSet<_> = d.task.splits["train"].iter().map(|r| &r.prompt).collect();
        for (split, rows) in &d.task.splits {
            for e in rows {
                assert_eq!(e.prompt.first(), Some(&1));
                assert_eq!(e.prompt.last(), Some(&18));
                assert_eq!(e.prompt.iter().filter(|&&t| t == 18).count(), 1);
                let word = &e.prompt[1..e.prompt.len() - 1];
                let distinct: BTreeSet<_> = word.iter().collect();
                assert!(word.iter().all(|&t| (36..100).contains(&t)));
                if case.ends_with("unique") {
                    assert_eq!(distinct.len(), word.len());
                } else {
                    assert!(distinct.len() <= if e.task == "sort" { 8 } else { 2 });
                }
                let expected = match e.task.as_str() {
                    "copy" => word.to_vec(),
                    "reverse" => (0..word.len()).map(|i| word[word.len() - 1 - i]).collect(),
                    "sort" => ranked(word),
                    _ => unreachable!(),
                };
                assert_eq!(e.answer[..word.len()], expected);
                assert_eq!(e.answer.last(), Some(&17));
                assert_eq!(e.limit, word.len() + 1);
                assert_eq!(
                    e.tokens,
                    e.prompt
                        .iter()
                        .chain(e.answer[..word.len()].iter())
                        .copied()
                        .collect::<Vec<_>>()
                );
                assert_eq!(
                    e.targets,
                    std::iter::repeat_n(-100, e.prompt.len() - 1)
                        .chain(e.answer.iter().copied())
                        .collect::<Vec<_>>()
                );
                assert!(e.changes[..e.prompt.len() - 1].iter().all(|c| !*c));
                for (i, &y) in e.answer.iter().enumerate() {
                    assert_eq!(
                        e.changes[e.prompt.len() - 1 + i],
                        i == 0 || e.answer[i - 1] != y
                    );
                }
                if split.ends_with("/novel") {
                    assert!(!train.contains(&e.prompt));
                }
            }
        }
        for (length, count) in [(32, 2112), (64, 4160)] {
            let rows = &d.task.splits[&format!("test/length-{length}")];
            assert_eq!(rows.iter().map(|e| e.answer.len()).sum::<usize>(), count);
            assert!(
                rows.iter()
                    .all(|e| e.tokens.len() == 2 * length + 2 && e.limit == length + 1)
            );
        }
        let meta = dataset::metadata(&d);
        assert_eq!(
            meta["generation"]["test"]["answer_tokens_including_eos"],
            targets
        );
    }
    Ok(())
}

#[test]
fn symbol_renaming_preserves_copy_and_reverse_and_sort_uses_ordinal_order() {
    let word = [40, 36, 40, 38];
    let renamed = [37, 41, 37, 39];
    for task in ["copy", "reverse", "sort"] {
        let original = generator::from_word(task, &word);
        let actual = generator::from_word(task, &renamed);
        let mapped: Vec<_> = original
            .answer
            .iter()
            .map(|&x| match x {
                40 => 37,
                36 => 41,
                38 => 39,
                17 => 17,
                _ => unreachable!(),
            })
            .collect();
        if task == "sort" {
            assert_eq!(actual.answer, vec![37, 37, 39, 41, 17]);
            assert_ne!(actual.answer, mapped);
        } else {
            assert_eq!(actual.answer, mapped);
        }
    }
}

#[test]
fn deterministic_stream_prefixes_and_bounds() -> Result<()> {
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
    spec["unique"] = json!(true);
    spec["symbols"] = json!(8);
    assert!(dataset::generate(&spec, "train", 8).is_err());
    assert!(dataset::build("suite-mqar").is_err());
    Ok(())
}

#[test]
fn generation_reads_its_own_mistake_and_teacher_forcing_is_distinct() -> Result<()> {
    let e = generator::from_word("copy", &[42, 43, 44]);
    let predict = |prefix: &[i64]| -> Result<usize> {
        Ok(match prefix.last() {
            Some(18) => 99,
            Some(42) => 43,
            Some(43) => 44,
            _ => 17,
        })
    };
    let forced: Vec<_> = (e.prompt.len() - 1..e.tokens.len())
        .map(|i| predict(&e.tokens[..=i]).unwrap())
        .collect();
    assert_eq!(forced, vec![99, 43, 44, 17]);
    let mut calls = vec![];
    let generated = scoring::rollout(&e.prompt, e.limit, 100, |prefix| {
        calls.push(prefix.to_vec());
        predict(prefix)
    })?;
    assert_eq!(generated, vec![99, 17]);
    assert_eq!(calls.len(), 2);
    assert_eq!(
        calls[1],
        e.prompt.iter().copied().chain([99]).collect::<Vec<_>>()
    );
    let score = scoring::score(&e.answer, &generated)?;
    assert_eq!(score.correct, 0);
    assert_eq!(score.targets, 4);
    assert!(!score.exact && score.ended);
    Ok(())
}

#[test]
fn wrong_early_missing_and_late_eos_keep_original_denominators() -> Result<()> {
    let answer = [42, 42, 43, 17];
    let exact = scoring::score(&answer, &[42, 42, 43, 17])?;
    assert_eq!(
        (
            exact.correct,
            exact.transition_correct,
            exact.transition_targets
        ),
        (4, 3, 3)
    );
    assert!(exact.exact && exact.ended);
    for (tokens, correct, ended, extra) in [
        (vec![42, 41, 43, 17], 3, true, 0),
        (vec![42, 17], 1, true, 0),
        (vec![42, 42, 43, 44], 3, false, 0),
        (vec![42, 42, 43, 44, 17], 3, true, 1),
    ] {
        let s = scoring::score(&answer, &tokens)?;
        assert_eq!(
            (s.targets, s.correct, s.ended, s.extra_tokens),
            (4, correct, ended, extra)
        );
        assert!(!s.exact);
    }
    assert!(scoring::score(&answer, &[42, 17, 43]).is_err());
    assert!(scoring::score(&answer, &[]).is_err());
    Ok(())
}

#[test]
fn eos_stops_and_limits_bound_actual_calls() -> Result<()> {
    let prompt = [1, 42, 18];
    let mut calls = 0;
    assert_eq!(
        scoring::rollout(&prompt, 4, 100, |_| {
            calls += 1;
            Ok(17)
        })?,
        vec![17]
    );
    assert_eq!(calls, 1);
    let mut lengths = vec![];
    assert_eq!(
        scoring::rollout(&prompt, 4, 100, |p| {
            lengths.push(p.len());
            Ok(42)
        })?,
        vec![42; 4]
    );
    assert_eq!(lengths, vec![3, 4, 5, 6]);
    assert!(scoring::rollout(&prompt, 4, 100, |_| Ok(100)).is_err());
    assert!(scoring::rollout(&prompt, 0, 100, |_| Ok(17)).is_err());
    // The largest admitted word has4096 symbols plus its EOS target.
    assert_eq!(
        scoring::rollout(&vec![42; 4098], 4097, 100, |_| Ok(17))?,
        vec![17]
    );
    assert!(scoring::rollout(&prompt, 4098, 100, |_| Ok(17)).is_err());
    Ok(())
}

#[test]
fn pinned_inputs_and_existing_output_refuse_changes() -> Result<()> {
    let root = std::env::temp_dir().join(format!("generative-data-pin-{}", std::process::id()));
    std::fs::create_dir(&root)?;
    let file = root.join("input.json");
    let bytes = b"{\"source\":1}\n";
    std::fs::write(&file, bytes)?;
    assert_eq!(
        inventory::read_pinned(&file, &hash(bytes))?,
        json!({"source":1})
    );
    std::fs::write(&file, b"{\"source\":2}\n")?;
    assert!(inventory::read_pinned(&file, &hash(bytes)).is_err());
    assert!(inventory::prepare(&file, &file, &file).is_err());
    std::fs::remove_dir_all(root)?;
    Ok(())
}

use crate::*;
use serde_json::json;
use std::collections::BTreeSet;

#[test]
fn all_cases_preserve_masks_lengths_limits_and_novelty() -> Result<()> {
    for case in CASES {
        let d = dataset::build(case)?;
        let (s, _) = dataset::specification(case)?;
        let vocab = if s["task"] == "crasp" {
            36
        } else if s["task"] == "random_lm" {
            105
        } else if s["task"] == "addition" && s["index_hints"] == false {
            100
        } else {
            229
        };
        assert_eq!(d.task.vocab, vocab);
        assert_eq!(
            d.task.splits["train"].len(),
            if s["task"] == "random_lm" { 8 } else { 128 }
        );
        assert_eq!(
            d.tensor_fingerprints["train"],
            d.tensor_fingerprints["train_clean"]
        );
        let training: BTreeSet<_> = d.task.splits["train"]
            .iter()
            .map(|e| dataset::key(e).to_vec())
            .collect();
        for (name, rows) in &d.task.splits {
            assert!(!rows.is_empty());
            for e in rows {
                assert_eq!(e.tokens.len(), e.targets.len());
                assert_eq!(e.tokens.len(), e.changes.len());
                assert!(e.tokens.iter().all(|&x| x >= 0 && (x as usize) < vocab));
                let mut previous = None;
                for (&y, &changed) in e.targets.iter().zip(&e.changes) {
                    assert!(y == -100 || (y >= 0 && (y as usize) < vocab));
                    assert_eq!(changed, y >= 0 && previous != Some(y));
                    if y >= 0 {
                        previous = Some(y);
                    }
                }
                if s["task"] == "crasp" {
                    assert_eq!(e.tokens.first(), Some(&1));
                    assert_eq!(e.targets.first(), Some(&-100));
                    assert!(e.targets[1..].iter().all(|&y| y == 15 || y == 16));
                    assert!(e.prompt.is_empty() && e.answer.is_empty() && e.limit == 0);
                } else {
                    assert_eq!(
                        e.tokens,
                        e.prompt
                            .iter()
                            .chain(&e.answer[..e.answer.len() - 1])
                            .copied()
                            .collect::<Vec<_>>()
                    );
                    assert_eq!(
                        e.targets,
                        std::iter::repeat_n(-100, e.prompt.len() - 1)
                            .chain(e.answer.iter().copied())
                            .collect::<Vec<_>>()
                    );
                    assert_eq!(e.answer.last(), Some(&17));
                    let expected = if s["task"] == "count" {
                        (e.prompt[2] - e.prompt[1] + 2) as usize
                    } else if s["task"] == "addition" {
                        addition::limit(&s, &e.prompt)?
                    } else {
                        (e.prompt[1] - 40 + 1) as usize
                    };
                    assert_eq!(e.limit, expected);
                    assert_eq!(e.answer.len(), expected);
                }
                if name.ends_with("/novel") {
                    assert!(!training.contains(dataset::key(e)));
                }
            }
        }
        let first = dataset::generate(&s, "train", 3)?;
        assert_eq!(
            wire::tensor_hash(&first),
            wire::tensor_hash(&d.task.splits["train"][..3])
        );
        assert_eq!(dataset::metadata(&d)?["new_model_evaluations"], 0);
    }
    Ok(())
}

#[test]
fn independent_digit_arithmetic_and_forced_carries_on_every_addition_split() -> Result<()> {
    for case in CASES.iter().filter(|c| c.contains("addition")) {
        let d = dataset::build(case)?;
        let (s, _) = dataset::specification(case)?;
        for (name, rows) in d.task.splits {
            for e in rows {
                let plus = e.prompt.iter().position(|&x| x == 20).unwrap();
                let hints = s["index_hints"] == true;
                let extract = |raw: &[i64]| {
                    raw.iter()
                        .enumerate()
                        .filter(|(i, _)| !hints || i % 2 == 1)
                        .map(|(_, t)| t - 26)
                        .collect::<Vec<_>>()
                };
                let a = extract(&e.prompt[1..plus]);
                let b = extract(&e.prompt[plus + 1..e.prompt.len() - 1]);
                let mut sum = vec![];
                let (mut carry, mut chain, mut longest) = (0, 0, 0);
                for (&a, &b) in a.iter().zip(&b).rev() {
                    let value = a + b + carry;
                    sum.push(value % 10);
                    carry = value / 10;
                    chain = if carry == 1 { chain + 1 } else { 0 };
                    longest = longest.max(chain);
                }
                assert_eq!(carry, 0);
                sum.reverse();
                if let Some(suffix) = name.split("hard-carry-length-").nth(1) {
                    assert_eq!(longest, suffix.split('/').next().unwrap().parse::<usize>()?);
                }
                let (parsed_a, parsed_b, indices) = addition::parse(&s, &e.prompt)?;
                assert_eq!(a, parsed_a);
                assert_eq!(b, parsed_b);
                let mut order: Vec<_> = (0..sum.len()).collect();
                if s["addition_order"] == "reverse" {
                    order.reverse();
                }
                let mut expected = vec![];
                for index in order {
                    if hints {
                        expected.push(indices[index]);
                    }
                    expected.push(sum[index] + 26);
                }
                expected.push(17);
                assert_eq!(e.answer, expected);
            }
        }
    }
    Ok(())
}

#[test]
fn addition_formats_share_problems_but_preserve_distinct_serializations() -> Result<()> {
    for carries in ["standard", "balanced"] {
        let (plain, _) =
            dataset::specification(&format!("suite-addition-forward-plain-{carries}"))?;
        let base = dataset::generate(&plain, "test", 65)?;
        for order in ["forward", "reverse"] {
            for hints in ["plain", "hints"] {
                let (s, _) =
                    dataset::specification(&format!("suite-addition-{order}-{hints}-{carries}"))?;
                let rows = dataset::generate(&s, "test", 65)?;
                for (a, b) in base.iter().zip(&rows) {
                    let (aa, ab, _) = addition::parse(&plain, &a.prompt)?;
                    let (ba, bb, _) = addition::parse(&s, &b.prompt)?;
                    assert_eq!((aa, ab), (ba, bb));
                }
                if hints == "hints" {
                    assert_ne!(wire::tensor_hash(&base), wire::tensor_hash(&rows));
                }
            }
        }
    }
    Ok(())
}

#[test]
fn formula_vectors_match_separate_recount_and_known_inclusive_examples() -> Result<()> {
    use crasp::{Formula as F, Term as T};
    let one = F::Lt(T::Count(Box::new(F::Sym(9))), T::Const(2));
    assert_eq!(one.evaluate(&[9, 10, 9])?, vec![true, true, false]);
    let two = F::Lt(T::Count(Box::new(one)), T::Const(2));
    assert_eq!(two.evaluate(&[9, 10, 9])?, vec![true, false, false]);
    assert_eq!(two.evaluate(&[9, 10, 9])?, two.independent(&[9, 10, 9])?);
    for depth in 1..=3 {
        let selected = crasp::select(depth, 0)?;
        assert_eq!(selected.formula.depth(), depth);
        for (index, word) in selected.witnesses.iter().enumerate() {
            assert_eq!(selected.formula.evaluate(word)?.last(), Some(&(index == 1)));
        }
        for length in 1..=5u32 {
            for index in 0..3usize.pow(length) {
                let mut rest = index;
                let word: Vec<_> = (0..length)
                    .map(|_| {
                        let s = crasp::SYMBOLS[rest % 3];
                        rest /= 3;
                        s
                    })
                    .collect();
                assert_eq!(
                    selected.formula.evaluate(&word)?,
                    selected.formula.independent(&word)?
                );
            }
        }
        let mut rng = Random::new(17);
        let long: Vec<_> = (0..64)
            .map(|_| crasp::SYMBOLS[rng.below(3) as usize])
            .collect();
        assert_eq!(
            selected.formula.evaluate(&long)?,
            selected.formula.independent(&long)?
        );
    }
    let overflow = F::Lt(
        T::Add(Box::new(T::Const(i64::MAX)), Box::new(T::Const(1))),
        T::Const(0),
    );
    assert!(overflow.evaluate(&[9]).is_err() && overflow.independent(&[9]).is_err());
    assert!(F::Sym(0).evaluate(&[9]).is_err());
    assert!(
        crasp::select(5, 0)?
            .formula
            .independent(&vec![9; 4096])
            .is_err()
    );
    Ok(())
}

#[test]
fn count_endpoints_and_random_payload_novelty_remain_distinct() -> Result<()> {
    let count = dataset::build("suite-count")?;
    for e in count.task.splits.values().flatten() {
        assert_eq!(e.prompt.len(), 4);
        let (start, end) = (e.prompt[1], e.prompt[2]);
        assert!(100 < start && start <= end && end <= 228);
        assert_eq!(
            e.answer[..e.answer.len() - 1],
            (start..=end).collect::<Vec<_>>()
        );
    }
    let d = dataset::build("random-lm-control")?;
    let train = &d.task.splits["train"];
    assert!(train.iter().all(|e| e.prompt == train[0].prompt));
    assert!(train.iter().any(|e| e.answer != train[0].answer));
    assert_ne!(dataset::key(&train[0]), dataset::key(&train[1]));
    assert!(d.task.splits.contains_key("validation/novel"));
    assert_eq!(dataset::metadata(&d)?["deterministic_target"], false);
    Ok(())
}

#[test]
fn scorers_keep_missing_extra_tokens_eos_and_original_prefix_masks() -> Result<()> {
    let answer = vec![101, 102, 17];
    let generated = scoring::rollout(&[1, 18], 3, 229, |p| {
        Ok(match p.last() {
            Some(18) => 111,
            Some(111) => 112,
            _ => 17,
        })
    })?;
    assert_eq!(generated, vec![111, 112, 17]);
    assert_eq!(scoring::score(&answer, &generated)?.correct, 1);
    assert!(!scoring::score(&answer, &[101, 102])?.exact);
    assert_eq!(scoring::score(&answer, &[101, 102, 0, 17])?.extra_tokens, 1);
    assert_eq!(scoring::score(&answer, &[17])?.correct, 0);
    let mut e = wire::example("crasp", vec![1, 9, 10, 9], vec![-100, 15, 16, 15]);
    let scored = scoring::prefix(&e, &[0, 15, 16, 15])?;
    assert!(scored.exact && scored.targets == 3 && scored.transition_targets == 3);
    e.changes[2] = false;
    assert!(scoring::prefix(&e, &[0, 15, 16, 15]).is_err());
    Ok(())
}

#[test]
fn invalid_shapes_formats_and_resource_bounds_refuse_before_sampling() -> Result<()> {
    let (s, _) = dataset::specification("suite-count")?;
    assert!(dataset::generate(&s, "test", 0).is_err());
    assert!(dataset::generate(&s, "bad", 64).is_err());
    for (field, value) in [
        ("length", json!(0)),
        ("length", json!(usize::MAX)),
        ("symbols", json!(0)),
        ("number_limit", json!(2)),
    ] {
        let mut bad = s.clone();
        bad[field] = value;
        assert!(dataset::generate(&bad, "test", 64).is_err());
    }
    let (mut s, _) = dataset::specification("suite-addition-forward-hints-balanced")?;
    s["carry_length"] = json!(3);
    assert!(dataset::generate(&s, "test", 64).is_err());
    let (mut s, _) = dataset::specification("suite-crasp-depth-1")?;
    s["formula_depth"] = json!(6);
    assert!(dataset::generate(&s, "test", 64).is_err());
    assert!(dataset::build("not-a-case").is_err());
    Ok(())
}

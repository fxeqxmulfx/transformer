use crate::*;
use serde_json::{Value, json};
use std::collections::BTreeSet;

fn spec(case: &str) -> Value {
    let p: Value = serde_json::from_str(PROTOCOL).unwrap();
    let mut out = p["definitions"][case].clone();
    out.as_object_mut().unwrap().remove("longer_lengths");
    out
}

#[test]
fn historical_denominators_and_independent_prefix_counts() -> Result<()> {
    let counts = [
        [(128, 128), (128, 128), (128, 128)],
        [(892, 489), (2176, 1135), (4224, 2163)],
        [(1656, 1656), (4224, 4224), (8320, 8320)],
        [(850, 850), (2142, 2142), (4198, 4198)],
    ];
    for (index, case) in CASES.iter().enumerate() {
        let d = dataset::build(case)?;
        assert_eq!(d.task.vocab, if index < 2 { 100 } else { 229 });
        assert_eq!(d.task.splits.len(), 11);
        assert_eq!(d.task.selection, "validation/novel");
        let train: BTreeSet<_> = d.task.splits["train"].iter().map(|e| &e.prompt).collect();
        for (name, rows) in &d.task.splits {
            for e in rows {
                assert_eq!(e.task, "parity");
                let body = &e.prompt[1..e.prompt.len() - 1];
                let bits: Vec<_> = body
                    .iter()
                    .copied()
                    .filter(|&x| x == 21 || x == 22)
                    .collect();
                let hints: Vec<_> = body.iter().copied().filter(|&x| x >= 100).collect();
                let mut reference = if index == 0 { vec![] } else { vec![24] };
                for i in 0..bits.len() {
                    if index == 0 && i + 1 != bits.len() || index == 3 && bits[i] != 22 {
                        continue;
                    }
                    if index >= 2 {
                        reference.push(hints[i]);
                    }
                    let ones = bits[..=i].iter().filter(|&&x| x == 22).count();
                    reference.push(24 + (ones % 2) as i64);
                }
                reference.push(17);
                assert_eq!(e.answer, reference);
                assert_eq!(e.prompt.first(), Some(&1));
                assert_eq!(e.prompt.last(), Some(&18));
                assert_eq!(
                    e.limit,
                    if index == 0 {
                        2
                    } else {
                        (if index >= 2 { 2 } else { 1 }) * bits.len() + 2
                    }
                );
                assert!(e.answer.len() <= e.limit);
                if index >= 2 {
                    assert_eq!(hints.len(), bits.len());
                    assert!(hints.windows(2).all(|w| w[1] == w[0] + 1));
                    assert!(hints[0] >= 100 && *hints.last().unwrap() < 228);
                    assert!(
                        body.as_chunks::<2>()
                            .0
                            .iter()
                            .all(|p| p[0] >= 100 && [21, 22].contains(&p[1]))
                    );
                }
                assert_eq!(
                    e.tokens,
                    e.prompt
                        .iter()
                        .chain(e.answer[..e.answer.len() - 1].iter())
                        .copied()
                        .collect::<Vec<_>>()
                );
                assert_eq!(
                    e.targets,
                    std::iter::repeat_n(-100, e.prompt.len() - 1)
                        .chain(e.answer.iter().copied())
                        .collect::<Vec<_>>()
                );
                assert!(e.changes[..e.prompt.len() - 1].iter().all(|c| !c));
                for (i, &y) in e.answer.iter().enumerate() {
                    assert_eq!(
                        e.changes[e.prompt.len() - 1 + i],
                        i == 0 || e.answer[i - 1] != y
                    );
                }
                if name.ends_with("/novel") {
                    assert!(!train.contains(&e.prompt));
                }
            }
        }
        for (split, (targets, transitions)) in ["test", "test/length-32", "test/length-64"]
            .into_iter()
            .zip(counts[index])
        {
            let rows = &d.task.splits[split];
            assert_eq!(rows.len(), 64);
            assert_eq!(rows.iter().map(|e| e.answer.len()).sum::<usize>(), targets);
            assert_eq!(
                rows.iter().flat_map(|e| &e.changes).filter(|&&c| c).count(),
                transitions
            );
        }
        assert_eq!(
            wire::tensor_hash(&d.task.splits["train"]),
            wire::tensor_hash(&d.task.splits["train_clean"])
        );
    }
    Ok(())
}

#[test]
fn formats_share_bits_and_hints_without_reseeding_problem_stream() -> Result<()> {
    for split in ["train", "validation", "test"] {
        let rows: Vec<_> = CASES
            .iter()
            .map(|case| dataset::generate(&spec(case), split, 17).unwrap())
            .collect();
        for i in 0..17 {
            let plain = &rows[0][i].prompt[1..rows[0][i].prompt.len() - 1];
            for format in &rows {
                let bits: Vec<_> = format[i]
                    .prompt
                    .iter()
                    .copied()
                    .filter(|&t| [21, 22].contains(&t))
                    .collect();
                assert_eq!(bits, plain);
            }
            assert_eq!(rows[2][i].prompt, rows[3][i].prompt);
        }
        let longer = dataset::generate(&spec(CASES[2]), split, 22)?;
        assert_eq!(
            wire::tensor_hash(&rows[2]),
            wire::tensor_hash(&longer[..17])
        );
    }
    assert!(dataset::identity(&spec(CASES[0]))["symbols"].is_null());
    assert_eq!(dataset::identity(&spec(CASES[2]))["symbols"], 64);
    Ok(())
}

#[test]
fn paired_draw_order_and_independent_format_stream() -> Result<()> {
    let s = spec(CASES[2]);
    let mut rng = wire::rng(&s, "test");
    let size = 8 + rng.below(9) as usize;
    let mut pair = vec![];
    for _ in 0..2 {
        let words: Vec<_> = (0..4).map(|_| rng.bits(32)).collect();
        let bits: Vec<_> = (0..size).map(|_| 21 + rng.below(2) as i64).collect();
        let start = 100 + Random::words(&words).below((129 - size) as u32) as usize;
        pair.push(generator::from_bits("running", Some(start), &bits));
    }
    if rng.random() < 0.5 {
        pair.reverse();
    }
    assert_eq!(
        wire::tensor_hash(&dataset::generate(&s, "test", 2)?),
        wire::tensor_hash(&pair)
    );
    assert_eq!(pair[0].prompt.len(), pair[1].prompt.len());
    Ok(())
}

#[test]
fn scratchpad_fixture_keeps_initial_state_and_original_one_bit_indices() {
    let bits = [21, 22, 21, 22, 22];
    assert_eq!(
        generator::from_bits("none", None, &bits).answer,
        vec![25, 17]
    );
    assert_eq!(
        generator::from_bits("running", None, &bits).answer,
        vec![24, 24, 25, 25, 24, 25, 17]
    );
    let ones = generator::from_bits("ones", Some(120), &bits);
    assert_eq!(ones.answer, vec![24, 121, 25, 123, 24, 124, 25, 17]);
    assert_eq!(ones.limit, 12);
    assert_eq!(
        generator::from_bits("ones", Some(100), &[21, 21]).answer,
        vec![24, 17]
    );
}

#[test]
fn own_output_generation_and_eos_keep_original_denominators() -> Result<()> {
    let row = generator::from_bits("running", Some(100), &[22, 21]);
    let mut seen = vec![];
    let answer = scoring::rollout(&row.prompt, row.limit, 229, |prefix| {
        seen.push(prefix.to_vec());
        Ok(if prefix.last() == Some(&18) { 25 } else { 17 })
    })?;
    assert_eq!(answer, vec![25, 17]);
    assert_eq!(seen[1].last(), Some(&25));
    let score = scoring::score(&row.answer, &answer)?;
    assert_eq!(
        (score.targets, score.correct, score.exact, score.ended),
        (6, 0, false, true)
    );
    let no_eos = scoring::rollout(&row.prompt, row.limit, 229, |_| Ok(24))?;
    assert_eq!(no_eos.len(), row.limit);
    assert!(!scoring::score(&row.answer, &no_eos)?.ended);
    Ok(())
}

#[test]
fn refusals_are_early_and_inputs_are_pinned() -> Result<()> {
    let mut s = spec(CASES[3]);
    assert!(dataset::generate(&s, "unknown", 8).is_err());
    assert!(dataset::generate(&s, "train", 0).is_err());
    assert!(dataset::generate(&s, "train", 20001).is_err());
    s["length"] = json!(129);
    assert!(dataset::generate(&s, "train", 8).is_err());
    s["length"] = json!(16);
    s["scratchpad"] = json!("unsupported");
    assert!(dataset::generate(&s, "train", 8).is_err());
    assert!(dataset::build("suite-copy-repeat").is_err());
    let root = std::env::temp_dir().join(format!("parity-pin-{}", std::process::id()));
    std::fs::create_dir(&root)?;
    let file = root.join("value.json");
    std::fs::write(&file, b"{}\n")?;
    assert!(inventory::read_pinned(&file, &hash(b"{}\n")).is_ok());
    assert!(inventory::read_pinned(&file, &hash(b"[]\n")).is_err());
    assert!(inventory::prepare(&file, &file, &file).is_err());
    std::fs::remove_dir_all(root)?;
    Ok(())
}

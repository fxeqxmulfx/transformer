use super::*;
use serde_json::json;
use std::{fs, path::Path};

#[test]
fn all_six_native_noise_draws_match_the_original_pcg_and_bounded_distributions() {
    let fixture = include_str!("../../../tests/reference/noisy_anchor_draws.tsv");
    for line in fixture.lines() {
        let values: Vec<_> = line.split_whitespace().collect();
        let population = values[0].parse().unwrap();
        let fraction = values[1].parse::<f64>().unwrap() / 100.;
        let count = values[2].parse::<usize>().unwrap();
        let draws = replacements(population, fraction, 1).unwrap();
        assert_eq!(draws.len(), count);
        let bytes: Vec<_> = draws
            .iter()
            .flat_map(|&(i, r)| (i as i64).to_le_bytes().into_iter().chain(r.to_le_bytes()))
            .collect();
        assert_eq!(arithmetic::hash(&bytes), values[3]);
    }
}

#[test]
fn official_prompt_order_masks_support_and_readouts_survive_each_training_noise_level() {
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.lake/reproduction/gpt");
    for (name, digest) in [
        (
            "src/diff_gpt/application/grokking/series_data.py",
            "8ae251246f274b74ebf3908e7e8939b60d3b9eb0aa158481314ea3bb1ef2911b",
        ),
        (
            "src/diff_gpt/application/grokking/recurrence_field_sampler.py",
            "8a9c620eb432184d08ca217b29f96a157d589c72a528ab2e24cc9ebfc53143cf",
        ),
        (
            "research/current_grokking/adapters.py",
            "4c36782edbb785432a7dfd71cf291acde2a1bf2c9337f27b5df854ec826cdbc2",
        ),
    ] {
        assert_eq!(
            arithmetic::hash(&fs::read(source.join(name)).unwrap()),
            digest
        );
    }
    for (dataset, order, n) in [("div97", 0, 4656), ("fibonacci", 1, 4704)] {
        let mut case = json!({"data_seed":1,"data":{"dataset":dataset,"order":order,"noise":0.0}});
        let (clean, baseline) = build(&case).unwrap();
        // The positive-order clean orbit and its original field decoder are also
        // reconstructed from the inverse transition and official codec in tests/temporal.rs.
        if order == 1 {
            let suite: Value = serde_json::from_str(arithmetic::ORIGINAL35).unwrap();
            let canonical = suite["tasks"]
                .as_array()
                .unwrap()
                .iter()
                .find(|r| r["name"] == "recurrence-fibonacci-k1-c2-lgt")
                .unwrap();
            let (mut full, fingerprints) = temporal::build(&canonical["case"]).unwrap();
            assert_eq!(fingerprints, canonical["data_fingerprints"]);
            assert_eq!(baseline["train"], fingerprints["train"]);
            let mut frequency = [0usize; 256];
            for row in &full.splits["train"] {
                for &token in row
                    .tokens
                    .iter()
                    .chain(row.targets.iter().filter(|&&x| x >= 0))
                {
                    frequency[token as usize] += 1;
                }
            }
            let mut retained = Vec::new();
            for mut row in full.splits.remove("validation").unwrap() {
                if [row.tokens[0], row.tokens[1], row.targets[1]]
                    .into_iter()
                    .all(|t| frequency[t as usize] > 0)
                {
                    row.row_index = Some((n + retained.len()) as i64);
                    retained.push(row);
                }
            }
            assert_eq!(retained.len(), 4700);
            assert_eq!(baseline["validation"], wire::tensor_hash(&retained));
        }
        let table = (order == 0).then(|| division(1).unwrap().1);
        for noise in [0.01, 0.05, 0.20] {
            case["data"]["noise"] = json!(noise);
            let (task, hashes) = build(&case).unwrap();
            assert_eq!(hashes["validation"], baseline["validation"]);
            let excluded = if order == 1 { 4 } else { 0 };
            assert_eq!(hashes["excluded_validation_rows"], excluded);
            let selected: BTreeMap<_, _> = replacements(n, noise, 1).unwrap().into_iter().collect();
            let mut changed = 0;
            let mut i = 0;
            for split in ["train", "validation"] {
                let required = if split == "validation" {
                    n - excluded
                } else {
                    n
                };
                assert_eq!(task.splits[split].len(), required);
                for (actual, original) in task.splits[split].iter().zip(&clean.splits[split]) {
                    assert_eq!(actual.tokens, original.tokens);
                    assert_eq!(actual.row_index, Some(i as i64));
                    assert_eq!(actual.targets.len(), 2);
                    assert_eq!(actual.targets[0], -100);
                    assert_eq!(actual.changes, [false, false]);
                    if let Some(&residue) = selected.get(&i) {
                        if let Some(table) = &table {
                            assert_eq!(actual.targets[1], table[residue as usize]);
                            assert!(actual.field.is_none());
                        } else {
                            let (base, _) = original.field.unwrap();
                            assert_eq!(actual.targets[1] + base - 128, residue);
                            assert_eq!(actual.field, Some((base, residue)));
                        }
                        changed += usize::from(actual.targets != original.targets);
                    } else {
                        assert_eq!(actual.targets, original.targets);
                        assert_eq!(actual.field, original.field);
                    }
                    i += 1;
                }
            }
            assert_eq!(hashes["selected_training_answers"], selected.len());
            assert_eq!(hashes["changed_training_answers"], changed);
        }
    }
}

//! Cross-check dividend copy against the complete, frozen divisor-control quartets.
//! The source shuffles the four query pairs within each unchanged memory.
use arithmetic::ORIGINAL35;
use basis::data::{Example, Task};
use serde_json::Value;
use std::collections::BTreeMap;
#[path = "../src/stress/associative_copy.rs"]
mod associative_copy;

fn reference() -> Value {
    let suite: Value = serde_json::from_str(ORIGINAL35).unwrap();
    suite["tasks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["name"] == "associative-divisor-n2-l1-w0")
        .unwrap()
        .clone()
}

fn swap_queries(mut data: Task) -> Task {
    for rows in data.splits.values_mut() {
        assert_eq!(rows.len(), 4656);
        let (groups, remainder) = rows.as_chunks_mut::<4>();
        assert!(remainder.is_empty());
        for group in groups {
            let memory = &group[0].tokens[..6];
            let responses: BTreeMap<_, _> = group
                .iter()
                .map(|example| {
                    assert_eq!(&example.tokens[..6], memory);
                    ((example.tokens[6], example.tokens[7]), example.targets[7])
                })
                .collect();
            assert_eq!(responses.len(), 4);
            for example in group {
                example.targets[7] = responses[&(example.tokens[7], example.tokens[6])];
            }
        }
    }
    data
}

fn changed(before: &[Example], after: &[Example]) -> usize {
    before
        .iter()
        .zip(after)
        .filter(|(a, b)| a.targets != b.targets)
        .count()
}

#[test]
fn swapping_the_operand_roles_in_every_frozen_quartet() {
    let control = reference();
    let original = basis::data::build(&control).unwrap();
    let swapped = swap_queries(basis::data::build(&control).unwrap());
    let mut case = control["case"].clone();
    case["data"]["labels"] = "dividend".into();
    let (actual_rows, hashes) = associative_copy::build(&case, &control).unwrap();
    for name in ["train", "validation"] {
        let (old, new) = (&original.splits[name], &swapped.splits[name]);
        let actual = basis::wire::tensor_hash(new);
        assert_eq!(
            basis::wire::tensor_hash(old),
            control["data_fingerprints"][name]
        );
        assert_eq!(hashes[name], actual);
        assert_eq!(changed(old, new), if name == "train" { 2304 } else { 2302 });
        for (row, expected) in actual_rows.splits[name].iter().zip(new) {
            assert_eq!(row.tokens, expected.tokens);
            assert_eq!(row.targets, expected.targets);
            assert_eq!(row.changes, expected.changes);
        }
        assert!(new.iter().zip(old).all(|(a, b)| a.tokens == b.tokens
            && a.targets[..7] == [-100; 7]
            && a.changes == b.changes));
    }
}

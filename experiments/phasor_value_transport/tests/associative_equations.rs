//! Original single-compute register frames, masks and independent source replay.
#[path = "../src/stress/associative_equations.rs"]
mod associative_equations;

use arithmetic::reference::SplitMix64;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::PathBuf,
};

fn settings(loads: usize, extra: usize, label: &str) -> Value {
    json!({"kind":"associative","data_seed":1,"data":{"compute":1,"loads":loads,"overwrites":extra,"labels":label}})
}

/// The browser's equation partition is independent of the random memory keys.
fn original_order() -> Vec<(i64, i64)> {
    let mut rows: Vec<_> = (0..97).flat_map(|a| (1..97).map(move |b| (a, b))).collect();
    SplitMix64(1 ^ 0x7370_6c69_7473_0002).shuffle(&mut rows);
    rows
}

/// Integer evaluation of the k=0 LGT scale used by the canonical divisor control.
fn token(level: i64) -> i64 {
    128 + (i128::from(level) * 6_004_799_503_160_661).div_euclid(1i128 << 52) as i64
}

#[test]
fn source_register_interpreter_checks_every_equation_mask_and_overwrite() {
    let equations = original_order();
    let levels: BTreeMap<_, _> = (-96..=96).map(|v| (token(v), v)).collect();
    for (loads, extra) in [(2, 0), (8, 0), (8, 4)] {
        let (task, _) = associative_equations::build(&settings(loads, extra, "division")).unwrap();
        for (name, pairs) in ["train", "validation"]
            .into_iter()
            .zip(equations.as_chunks::<4656>().0.iter())
        {
            let rows = &task.splits[name];
            assert_eq!(rows.len(), 4656);
            for (ex, &pair) in rows.iter().zip(pairs) {
                let (writes, queries) = ex.tokens.split_at(3 * (loads + extra));
                let mut memory = BTreeMap::new();
                for write in writes.as_chunks::<3>().0 {
                    let (key, marker, value) =
                        (levels[&write[0]], levels[&write[1]], levels[&write[2]]);
                    assert!((-32..=-1).contains(&key) && marker == -96 && (0..97).contains(&value));
                    assert_ne!(memory.insert(key, value), Some(value));
                }
                assert_eq!(memory.len(), loads);
                assert_eq!(queries.len(), 2);
                assert_ne!(queries[0], queries[1]);
                assert_eq!(
                    (memory[&levels[&queries[0]]], memory[&levels[&queries[1]]]),
                    pair
                );
                let (y, mask) = ex.targets.split_last().unwrap();
                assert_eq!(ex.tokens.len(), ex.targets.len());
                assert!(mask.iter().all(|&x| x == -100));
                assert_eq!((pair.1 * levels[y]) % 97, pair.0);
                assert!(ex.changes.iter().all(|&c| !c) && ex.row_index.is_none());
            }
        }
    }
    let div = associative_equations::build(&settings(2, 0, "division"))
        .unwrap()
        .0;
    let random = associative_equations::build(&settings(2, 0, "random"))
        .unwrap()
        .0;
    for name in ["train", "validation"] {
        let mut outputs = BTreeSet::new();
        let mut changed = 0;
        for (a, b) in div.splits[name].iter().zip(&random.splits[name]) {
            assert_eq!(a.tokens, b.tokens);
            assert_eq!(a.targets[..7], b.targets[..7]);
            assert_eq!(a.changes, b.changes);
            outputs.insert(b.targets[7]);
            changed += usize::from(a.targets[7] != b.targets[7]);
        }
        assert_eq!(outputs.len(), 97);
        assert!(changed > 4500);
    }
    assert!(associative_equations::build(&settings(2, 0, "dividend")).is_err());
}

#[test]
#[ignore = "requires frames exported by the independent native NumPy reference"]
fn every_context_matches_numpy_c_and_the_pinned_source_codec() {
    let folder = PathBuf::from(
        std::env::var_os("PHASOR_EQUATION_REFERENCE").expect("native reference folder"),
    );
    for (loads, extra, random) in [(2, 0, 0), (2, 0, 1), (8, 0, 0), (8, 4, 0)] {
        let data = settings(
            loads,
            extra,
            if random == 1 { "random" } else { "division" },
        );
        let (task, _) = associative_equations::build(&data).unwrap();
        for (i, split) in ["train", "validation"].into_iter().enumerate() {
            let file = folder.join(format!("frames-{loads}-{extra}-{i}-{random}.i16"));
            let bytes = fs::read(file).unwrap();
            let raw: Vec<_> = bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| i64::from(i16::from_le_bytes([b[0], b[1]])))
                .collect();
            let len = 3 * (loads + extra) + 3;
            let examples = &task.splits[split];
            assert_eq!(raw.len(), examples.len() * len);
            for (row, ex) in raw.chunks_exact(len).zip(examples) {
                assert!(
                    ex.tokens
                        .iter()
                        .copied()
                        .eq(row[..len - 1].iter().copied().map(token))
                );
                assert_eq!(ex.targets[len - 2], token(row[len - 1]));
                assert!(ex.targets[..len - 2].iter().all(|&v| v == -100));
            }
        }
    }
}

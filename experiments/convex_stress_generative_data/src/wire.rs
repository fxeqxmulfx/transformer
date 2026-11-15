//! Exact JSON and tensor hashing from convex_stress/vendor/data.rs at07a39ac2.
//! RNG identity follows synthetic/splits.py and records follow records.py;
//! these Python sources are frozen in task318 and are never executed here.
use crate::{Random, data::Example, hash};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

pub(crate) fn example(task: &str, tokens: Vec<i64>, targets: Vec<i64>) -> Example {
    let mut previous = None;
    let changes = targets
        .iter()
        .map(|&target| {
            let changed = target >= 0 && previous != Some(target);
            if target >= 0 {
                previous = Some(target);
            }
            changed
        })
        .collect();
    Example {
        task: task.into(),
        tokens,
        targets,
        changes,
        prompt: vec![],
        answer: vec![],
        limit: 0,
        field: None,
        row_index: None,
    }
}

pub fn value(e: &Example) -> Value {
    json!({"task":e.task,"tokens":e.tokens,"targets":e.targets,"changes":e.changes,
        "prompt":e.prompt,"answer":e.answer,"generation_limit":e.limit})
}

pub fn tensor_hash(examples: &[Example]) -> String {
    let width = examples.iter().map(|e| e.tokens.len()).max().unwrap_or(0);
    let mut bytes = format!("torch.int64:({}, 3, {}):", examples.len(), width).into_bytes();
    for e in examples {
        for part in 0..3 {
            for col in 0..width {
                let x = match part {
                    0 => e.tokens.get(col).copied().unwrap_or(0),
                    1 => e.targets.get(col).copied().unwrap_or(-100),
                    _ => i64::from(e.changes.get(col).copied().unwrap_or(false)),
                };
                bytes.extend_from_slice(&x.to_le_bytes());
            }
        }
    }
    hash(&bytes)
}

pub fn split_hash(spec: &Value, split: &str, examples: &[Example]) -> String {
    hash(
        crate::data::json_text(&json!({"version":2,"name":split,"spec":spec,"seed":1,
        "examples":examples.iter().map(value).collect::<Vec<_>>()}))
        .as_bytes(),
    )
}

pub(crate) fn rng(spec: &Value, split: &str) -> Random {
    let bytes = Sha256::digest(
        crate::data::json_text(&json!({"version":2,"seed":1,
        "split":split,"spec":spec}))
        .as_bytes(),
    );
    let words: Vec<_> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .rev()
        .map(|x| u32::from_be_bytes(*x))
        .collect();
    Random::words(&words)
}

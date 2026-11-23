//! Port of task340 wire.rs at28dc5c94, with parity format exclusion added.
//! Source splits.py/records.py JSON, Python integer-seeded MT19937 and tensors.
//! Python sources are read as references only; no Python process is invoked.
use crate::*;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

pub fn example(task: &str, tokens: Vec<i64>, targets: Vec<i64>) -> data::Example {
    let mut previous = None;
    let changes = targets
        .iter()
        .map(|&y| {
            let changed = y >= 0 && previous != Some(y);
            if y >= 0 {
                previous = Some(y);
            }
            changed
        })
        .collect();
    data::Example {
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
pub fn answered(task: &str, prompt: Vec<i64>, answer: Vec<i64>, limit: usize) -> data::Example {
    let tokens = prompt
        .iter()
        .chain(&answer[..answer.len() - 1])
        .copied()
        .collect();
    let targets = std::iter::repeat_n(-100, prompt.len() - 1)
        .chain(answer.iter().copied())
        .collect();
    let mut e = example(task, tokens, targets);
    e.limit = limit;
    e.prompt = prompt;
    e.answer = answer;
    e
}
pub fn value(e: &data::Example) -> Value {
    json!({"task":e.task,"tokens":e.tokens,"targets":e.targets,"changes":e.changes,
        "prompt":e.prompt,"answer":e.answer,"generation_limit":e.limit})
}
pub fn split_hash(s: &Value, split: &str, rows: &[data::Example]) -> String {
    hash(
        data::json_text(&json!({"version":2,"name":split,"spec":s,"seed":1,
        "examples":rows.iter().map(value).collect::<Vec<_>>()}))
        .as_bytes(),
    )
}
pub fn tensor_hash(rows: &[data::Example]) -> String {
    let width = rows.iter().map(|e| e.tokens.len()).max().unwrap_or(0);
    let mut bytes = format!("torch.int64:({}, 3, {}):", rows.len(), width).into_bytes();
    for e in rows {
        for part in 0..3 {
            for col in 0..width {
                let x = match part {
                    0 => e.tokens.get(col).copied().unwrap_or(0),
                    1 => e.targets.get(col).copied().unwrap_or(-100),
                    _ => match (col, e.row_index) {
                        (0, Some(index)) => index,
                        _ => i64::from(e.changes.get(col).copied().unwrap_or(false)),
                    },
                };
                bytes.extend_from_slice(&x.to_le_bytes());
            }
        }
    }
    hash(&bytes)
}
fn seeded(mut words: Vec<u32>) -> Random {
    // Python integer seeds omit leading zero words, including a 128-bit draw.
    while words.len() > 1 && words.last() == Some(&0) {
        words.pop();
    }
    Random::words(&words)
}
pub fn format_rng(rng: &mut Random) -> Random {
    seeded((0..4).map(|_| rng.next_u32()).collect())
}
pub fn addressed_rng(identity: &Value) -> Random {
    let bytes = Sha256::digest(data::json_text(identity).as_bytes());
    seeded(
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .rev()
            .map(|w| u32::from_be_bytes(*w))
            .collect(),
    )
}
pub fn rng(s: &Value, split: &str) -> Random {
    let mut problem = s.clone();
    let formats: &[&str] = match s["task"].as_str() {
        Some("parity") => &["scratchpad", "index_hints", "symbols", "number_limit"],
        _ => &[],
    };
    for key in formats {
        problem.as_object_mut().unwrap().remove(*key);
    }
    addressed_rng(&json!({"version":2,"seed":1,"split":split,"spec":problem}))
}

//! Original records.py JSON and data.py tensor fingerprints at DiffGPT5a37c515.
//! Numeric teachers retain every binary64 word, independently of quantized labels.
use crate::{Result, data, hash};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Record {
    pub task: String,
    pub tokens: Vec<i64>,
    pub targets: Vec<i64>,
    pub changes: Vec<bool>,
    pub prompt: Vec<i64>,
    pub answer: Vec<i64>,
    pub generation_limit: usize,
}
impl Record {
    pub fn generation(task: &str, input: &[i64], answer: &[i64]) -> Self {
        let prompt: Vec<_> = std::iter::once(1)
            .chain(input.iter().copied())
            .chain([18])
            .collect();
        let answer: Vec<_> = answer.iter().copied().chain([17]).collect();
        let tokens = prompt
            .iter()
            .chain(&answer[..answer.len() - 1])
            .copied()
            .collect();
        let targets: Vec<_> = std::iter::repeat_n(-100, prompt.len() - 1)
            .chain(answer.iter().copied())
            .collect();
        let mut previous = None;
        let changes = targets
            .iter()
            .map(|&y| {
                let changed = y != -100 && previous != Some(y);
                if y != -100 {
                    previous = Some(y);
                }
                changed
            })
            .collect();
        let generation_limit = answer.len();
        Self {
            task: task.into(),
            tokens,
            targets,
            changes,
            prompt,
            answer,
            generation_limit,
        }
    }
    pub fn from_original(e: &data::Example) -> Self {
        Self {
            task: e.task.clone(),
            tokens: e.tokens.clone(),
            targets: e.targets.clone(),
            changes: e.changes.clone(),
            prompt: e.prompt.clone(),
            answer: e.answer.clone(),
            generation_limit: e.limit,
        }
    }
    pub fn validate(&self, vocabulary: usize) -> Result<()> {
        let n = self.tokens.len();
        if n == 0
            || n > 8195
            || self.targets.len() != n
            || self.changes.len() != n
            || self
                .tokens
                .iter()
                .chain(&self.prompt)
                .chain(&self.answer)
                .any(|&t| t < 0 || t as usize >= vocabulary)
            || self
                .targets
                .iter()
                .any(|&t| t != -100 && (t < 0 || t as usize >= vocabulary))
            || !self.targets.iter().any(|&t| t >= 0)
        {
            return Err("invalid original probe record".into());
        }
        if !self.prompt.is_empty() {
            if self.answer.is_empty()
                || self.answer.last() != Some(&17)
                || self.answer[..self.answer.len() - 1].contains(&17)
                || self.generation_limit != self.answer.len()
                || self.tokens
                    != self
                        .prompt
                        .iter()
                        .chain(&self.answer[..self.answer.len() - 1])
                        .copied()
                        .collect::<Vec<_>>()
                || self.targets
                    != std::iter::repeat_n(-100, self.prompt.len() - 1)
                        .chain(self.answer.iter().copied())
                        .collect::<Vec<_>>()
            {
                return Err("original free-generation masks or EOS changed".into());
            }
        } else if !self.answer.is_empty() || self.generation_limit != 0 {
            return Err("dense probe has spurious generation data".into());
        }
        Ok(())
    }
}

pub fn examples_hash(records: &[Record]) -> Result<String> {
    Ok(hash(
        data::json_text(&serde_json::to_value(records)?).as_bytes(),
    ))
}
pub fn binding_hash(name: &str, records: &[Record]) -> String {
    if name.ends_with("/novel") {
        return tensor_hash(records);
    }
    let spec = json!({"task":"mqar","length":12,"min_length":null,"symbols":32,
        "pairs":2,"queries":1,"query_gap":0,"alpha":0.1});
    hash(
        data::json_text(&json!({"version":2,"name":name,"spec":spec,"seed":1,
        "examples":records}))
        .as_bytes(),
    )
}
pub fn tensor_hash(records: &[Record]) -> String {
    let width = records.iter().map(|e| e.tokens.len()).max().unwrap_or(0);
    let mut bytes = format!("torch.int64:({}, 3, {}):", records.len(), width).into_bytes();
    for e in records {
        for part in 0..3 {
            for i in 0..width {
                let value = match part {
                    0 => e.tokens.get(i).copied().unwrap_or(0),
                    1 => e.targets.get(i).copied().unwrap_or(-100),
                    _ => i64::from(e.changes.get(i).copied().unwrap_or(false)),
                };
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
    }
    hash(&bytes)
}
pub fn teacher_hash(words: &[Vec<u64>]) -> Result<String> {
    let width = words.first().ok_or("empty numeric teacher")?.len();
    if width == 0
        || words
            .iter()
            .any(|row| row.len() != width || row.iter().any(|&x| !f64::from_bits(x).is_finite()))
    {
        return Err("invalid original numeric teacher".into());
    }
    let mut bytes = format!("torch.float64:({}, {}):", words.len(), width).into_bytes();
    for row in words {
        for x in row {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
    }
    Ok(hash(&bytes))
}
pub fn summary(records: &[Record], teacher: Option<&[Vec<u64>]>) -> Result<Value> {
    Ok(
        json!({"examples":records.len(),"targets":records.iter().flat_map(|r| &r.targets).filter(|&&t|t>=0).count(),
        "record_sha256":examples_hash(records)?,"tensor_sha256":tensor_hash(records),
        "teacher_sha256":teacher.map(teacher_hash).transpose()?}),
    )
}

//! Original modular-composition sampling and final-answer masks. Only the
//! resulting raw object prefixes and final labels are needed by a learner.
use crate::Result;
use baseline::{
    arithmetic_data::SplitMix64,
    data::{Example, Task},
    hash,
};
use serde::Serialize;
use std::collections::{BTreeMap, HashSet};

pub const SOURCES: &str = include_str!("../chain_data/sources.json");
pub const PRIMES: [usize; 3] = [17, 97, 193];
pub const STEPS: [usize; 4] = [1, 2, 4, 8];

fn index(random: &mut SplitMix64, n: usize) -> i64 {
    let bound = n as u64;
    let threshold = bound.wrapping_neg() % bound;
    loop {
        let x = random.next();
        if x >= threshold {
            return (x % bound) as i64;
        }
    }
}

// Follow unique_inputs in the original source, including the small-domain
// threshold, the least significant digit first, and the order of accepted rows.
fn inputs(prime: usize, length: usize, count: usize, seed: u64) -> Vec<Vec<i64>> {
    let mut random = SplitMix64(seed);
    let domain = prime.saturating_pow(length as u32);
    if domain < 2 * count {
        let mut codes: Vec<_> = (0..domain).collect();
        random.shuffle(&mut codes);
        return codes
            .into_iter()
            .take(count)
            .map(|mut code| {
                (0..length)
                    .map(|_| {
                        let digit = code % prime;
                        code /= prime;
                        digit as i64
                    })
                    .collect()
            })
            .collect();
    }
    let mut rows = Vec::with_capacity(count);
    let mut seen = HashSet::with_capacity(count);
    while rows.len() < count {
        let row: Vec<_> = (0..length).map(|_| index(&mut random, prime)).collect();
        if seen.insert(row.clone()) {
            rows.push(row);
        }
    }
    rows
}

fn examples(inputs: Vec<Vec<i64>>, prime: usize) -> Vec<Example> {
    inputs
        .into_iter()
        .map(|tokens| {
            let mut z = tokens[0];
            for &[b, c] in tokens[1..].as_chunks::<2>().0 {
                z = (z.pow(3) + b * z + c) % prime as i64;
            }
            let mut targets = vec![-100; tokens.len() - 1];
            targets.push(z);
            Example {
                task: "chain".into(),
                changes: vec![false; tokens.len()],
                tokens,
                targets,
                prompt: Vec::new(),
                answer: Vec::new(),
                limit: 0,
                field: None,
                row_index: None,
            }
        })
        .collect()
}

pub fn build(prime: usize, steps: usize, data_seed: u64) -> Result<Task> {
    if !PRIMES.contains(&prime) || !STEPS.contains(&steps) {
        return Err("outside the original twelve modular-chain cases".into());
    }
    let mut short = inputs(
        prime,
        1 + 2 * steps,
        3072,
        data_seed ^ 0x7370_6c69_7473_0002,
    );
    let validation = short.split_off(2048);
    let longer = inputs(
        prime,
        1 + 4 * steps,
        1024,
        data_seed ^ 0x6c6f_6e67_6572_0001,
    );
    Ok(Task {
        vocab: prime,
        selection: "validation".into(),
        splits: BTreeMap::from([
            ("train".into(), examples(short, prime)),
            ("validation".into(), examples(validation, prime)),
            ("test/length-double".into(), examples(longer, prime)),
        ]),
    })
}

#[derive(Serialize, PartialEq, Eq, Debug)]
pub struct Split {
    pub examples: usize,
    pub input_length: usize,
    pub input_fingerprint: String,
    pub fingerprint: String,
    pub dense_tensor_sha256: String,
}

pub fn fingerprints(rows: &[Example], vocabulary: usize) -> Result<Split> {
    let width = rows
        .first()
        .ok_or("empty original chain split")?
        .tokens
        .len();
    if width == 0 {
        return Err("empty original chain prefix".into());
    }
    let mut inputs = Vec::new();
    let mut labeled = Vec::new();
    let mut dense = format!("torch.int64:({}, 3, {}):", rows.len(), width).into_bytes();
    let valid = |&x: &i64| (0..vocabulary as i64).contains(&x);
    for r in rows {
        if r.tokens.len() != width
            || !r.tokens.iter().all(valid)
            || r.targets.len() != width
            || !r.targets[..width - 1].iter().all(|&x| x == -100)
            || !valid(&r.targets[width - 1])
            || r.changes.len() != width
            || r.changes.iter().any(|&x| x)
            || r.row_index.is_some()
            || r.field.is_some()
            || r.limit != 0
            || !r.prompt.is_empty()
            || !r.answer.is_empty()
        {
            return Err("invalid original modular-chain row or final-only mask".into());
        }
        for x in &r.tokens {
            let bytes = x.to_le_bytes();
            inputs.extend(bytes);
            labeled.extend(bytes);
            dense.extend(bytes);
        }
        labeled.extend(r.targets[width - 1].to_le_bytes());
        dense.extend(r.targets.iter().flat_map(|x| x.to_le_bytes()));
        dense.resize(dense.len() + 8 * width, 0);
    }
    Ok(Split {
        examples: rows.len(),
        input_length: width,
        input_fingerprint: hash(&inputs),
        fingerprint: hash(&labeled),
        dense_tensor_sha256: hash(&dense),
    })
}

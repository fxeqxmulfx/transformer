//! Fixed-input tasks. Labels are constructed here, never inside a forward pass.

use crate::mt::Mt;
use sha2::{Digest, Sha256};
use std::collections::HashSet;

#[derive(Clone, Debug)]
pub struct Sample {
    pub tokens: Vec<usize>,
    pub targets: Vec<usize>,
    /// Dictionary values in key order, or [a,b] for modular division.
    pub values: Vec<usize>,
    pub queries: Vec<usize>,
    pub id: usize,
}

pub struct Dataset {
    pub train: Vec<Sample>,
    pub test: Vec<Sample>,
    pub vocab: usize,
    pub ctx: usize,
    pub sup: Vec<usize>,
    pub keys: usize,
    pub p: usize,
}

/// Complete dictionaries with independently shuffled binding and query orders.
/// All keys are queried; test dictionaries never occur in the training set.
pub fn mqar(seed: u32, keys: usize, values: usize, train: usize, test: usize) -> Dataset {
    let mut rng = Mt::new(seed.wrapping_add(17_003));
    let mut seen = HashSet::new();
    let mut samples = Vec::new();
    while samples.len() < train + test {
        let dict: Vec<_> = (0..keys).map(|_| rng.interval(values as u32 - 1) as usize).collect();
        if !seen.insert(dict.clone()) {
            continue;
        }
        let mut order: Vec<_> = (0..keys).collect();
        let mut queries = order.clone();
        rng.shuffle(&mut order);
        rng.shuffle(&mut queries);
        let mut tokens = Vec::new();
        for &k in &order {
            tokens.extend([k, keys + dict[k]]);
        }
        tokens.extend(&queries);
        let targets = queries.iter().map(|&q| keys + dict[q]).collect();
        samples.push(Sample { tokens, targets, values: dict, queries, id: samples.len() });
    }
    let held = samples.split_off(train);
    Dataset {
        train: samples,
        test: held,
        vocab: keys + values,
        ctx: 3 * keys,
        sup: (2 * keys..3 * keys).collect(),
        keys,
        p: 0,
    }
}

fn pow_mod(mut a: usize, mut n: usize, p: usize) -> usize {
    let mut out = 1;
    while n > 0 {
        if n & 1 != 0 {
            out = out * a % p;
        }
        a = a * a % p;
        n >>= 1;
    }
    out
}

/// A uniformly shuffled 50/50 split of all a/b mod p with nonzero b.
/// Inputs contain only [a, division-token, b, equals-token].
pub fn division(seed: u32, p: usize) -> Dataset {
    assert!(p >= 3 && (2..p).all(|d| !p.is_multiple_of(d)), "p must be an odd prime");
    let mut samples = Vec::new();
    for a in 0..p {
        for b in 1..p {
            let answer = a * pow_mod(b, p - 2, p) % p;
            samples.push(Sample {
                tokens: vec![a, p, b, p + 1],
                targets: vec![answer],
                values: vec![a, b],
                queries: vec![a],
                id: a * (p - 1) + b - 1,
            });
        }
    }
    Mt::new(seed).shuffle(&mut samples);
    let test = samples.split_off(samples.len() / 2);
    Dataset { train: samples, test, vocab: p + 2, ctx: 4, sup: vec![3], keys: 0, p }
}

/// Check the Lean theorem's separation condition using training inputs only.
pub fn separates(data: &Dataset) -> bool {
    (0..data.keys)
        .all(|q| (0..data.keys).all(|k| q == k || data.train.iter().any(|s| s.values[q] != s.values[k])))
}

pub fn fingerprint(samples: &[Sample]) -> String {
    let mut hash = Sha256::new();
    for s in samples {
        for &x in s.tokens.iter().chain(&s.targets) {
            hash.update((x as u64).to_le_bytes());
        }
    }
    hash.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// Upper bound for every rule constrained to copy a numeric input token.
pub fn copy_support_accuracy(samples: &[Sample]) -> f64 {
    samples.iter().filter(|s| s.values.contains(&s.targets[0])).count() as f64 / samples.len() as f64
}

//! Port of frozen synthetic/retrieval.py: original MQAR and paired two-hop
//! cyclic lookup distributions. Task318 pins the source and all defaults.
use crate::{Random, data::Example, wire};
use serde_json::Value;
use std::collections::BTreeMap;

pub(crate) fn mqar(spec: &Value, rng: &mut Random, length: usize) -> Example {
    let symbols = spec["symbols"].as_u64().unwrap() as usize;
    let pairs = spec["pairs"].as_u64().unwrap() as usize;
    let queries = spec["queries"].as_u64().unwrap() as usize;
    let keys = rng.sample(symbols, pairs);
    let values: Vec<_> = (0..pairs)
        .map(|_| 36 + symbols as i64 + rng.below(symbols as u32) as i64)
        .collect();
    let mut tokens = vec![1];
    tokens.extend((1..length).map(|_| 36 + symbols as i64 + rng.below(symbols as u32) as i64));
    for i in 0..pairs {
        tokens[1 + 2 * i] = 36 + keys[i] as i64;
        tokens[2 + 2 * i] = values[i];
    }
    let gap = spec["query_gap"].as_u64().unwrap() as usize;
    let alpha = spec["alpha"].as_f64().unwrap();
    let mut scores: Vec<_> = (1 + 2 * pairs + gap..length)
        .map(|p| {
            (
                (-rng.random().max(f64::MIN_POSITIVE).ln()).ln() + alpha * (p as f64).ln(),
                p,
            )
        })
        .collect();
    scores.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let selected = rng.sample(pairs, queries);
    let mut targets = vec![-100; length];
    for ((_, pos), key) in scores.iter().take(queries).zip(selected) {
        tokens[*pos] = 36 + keys[key] as i64;
        targets[*pos] = values[key];
    }
    wire::example("mqar", tokens, targets)
}

pub(crate) fn lookup(spec: &Value, rng: &mut Random, length: usize) -> Vec<Example> {
    let symbols = spec["symbols"].as_u64().unwrap() as usize;
    let pairs = spec["pairs"].as_u64().unwrap() as usize;
    let queries = spec["queries"].as_u64().unwrap() as usize;
    let hops = spec["hops"].as_u64().unwrap() as usize;
    let gap = spec["query_gap"].as_u64().unwrap() as usize;
    let identities: Vec<_> = rng
        .sample(symbols, pairs)
        .into_iter()
        .map(|x| 36 + x as i64)
        .collect();
    let cycle: Vec<_> = rng
        .sample(pairs, pairs)
        .into_iter()
        .map(|i| identities[i])
        .collect();
    let table: BTreeMap<_, _> = cycle
        .iter()
        .enumerate()
        .map(|(i, &k)| (k, cycle[(i + 1) % pairs]))
        .collect();
    let rows: Vec<_> = rng
        .sample(pairs, pairs)
        .into_iter()
        .map(|i| identities[i])
        .collect();
    let starts: Vec<_> = rng
        .sample(pairs, queries)
        .into_iter()
        .map(|i| identities[i])
        .collect();
    let mut gaps = wire::distribute(
        length - (2 + 4 * pairs + 2 * queries) - gap,
        queries + 1,
        rng,
    );
    gaps[0] += gap;
    let encode = |map: &BTreeMap<i64, i64>| {
        let mut tokens = vec![1];
        for &key in &rows {
            tokens.extend([2, key, 3, map[&key]]);
        }
        tokens.push(5);
        tokens.extend(std::iter::repeat_n(6, gaps[0]));
        let mut targets = vec![-100; tokens.len()];
        for (&start, &gap) in starts.iter().zip(&gaps[1..]) {
            let mut value = start;
            for _ in 0..hops {
                value = map[&value];
            }
            tokens.extend([4, start]);
            targets.extend([-100, value]);
            tokens.extend(std::iter::repeat_n(6, gap));
            targets.extend(std::iter::repeat_n(-100, gap));
        }
        wire::example("lookup", tokens, targets)
    };
    let first = encode(&table);
    if pairs == 2 {
        return vec![first.clone(), first];
    }
    let start = starts[0];
    let mut end = start;
    for _ in 0..hops {
        end = table[&end];
    }
    let alternatives: Vec<_> = identities
        .iter()
        .copied()
        .filter(|&x| x != start && x != end)
        .collect();
    let alternative = alternatives[rng.below(alternatives.len() as u32) as usize];
    let swap = |x| {
        if x == end {
            alternative
        } else if x == alternative {
            end
        } else {
            x
        }
    };
    let changed = table.iter().map(|(&k, &v)| (swap(k), swap(v))).collect();
    vec![first, encode(&changed)]
}

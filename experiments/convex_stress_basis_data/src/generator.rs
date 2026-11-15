//! Source draw order: synthetic/base.py, prefixes.py and retrieval.py frozen
//! by task318. Rewrites exclude the previous value; pair reversal is last.
use crate::*;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

pub fn number(s: &Value, key: &str) -> Result<usize> {
    usize::try_from(
        s[key]
            .as_u64()
            .ok_or("missing unsigned specification field")?,
    )
    .map_err(Into::into)
}
fn distribute(total: usize, parts: usize, rng: &mut Random) -> Vec<usize> {
    let mut out = vec![0; parts];
    for _ in 0..total {
        out[rng.below(parts as u32) as usize] += 1;
    }
    out
}
fn blocks_row(word: &[i64], gaps: &[usize], blocks: usize) -> data::Example {
    let mut tokens = vec![1];
    tokens.extend(std::iter::repeat_n(11, gaps[0]));
    for (t, gap) in word.iter().zip(&gaps[1..]) {
        tokens.push(*t);
        tokens.extend(std::iter::repeat_n(11, *gap));
    }
    let (mut first, mut previous, mut runs) = (0, 0, 0);
    let mut labels = vec![-100];
    for &t in &tokens[1..] {
        if t != 11 {
            if first == 0 {
                first = t;
            }
            runs += usize::from(t != previous);
            previous = t;
        }
        labels.push(if first == 9 && runs == blocks { 16 } else { 15 });
    }
    wire::example("blocks", tokens, labels)
}
fn blocks_pair(s: &Value, rng: &mut Random, length: usize) -> Vec<data::Example> {
    let blocks = s["blocks"].as_u64().unwrap() as usize;
    let size = ((length - 1) as f64 * (1. - s["neutral_fraction"].as_f64().unwrap()))
        .round_ties_even() as usize;
    if rng.random() < 1. / 3. {
        return (0..2)
            .map(|_| {
                let runs = 1 + rng.below(size.min(2 * blocks + 3) as u32) as usize;
                let counts = distribute(size - runs, runs, rng);
                let start = 9 + rng.below(2) as i64;
                let word: Vec<_> = counts
                    .into_iter()
                    .enumerate()
                    .flat_map(|(i, n)| {
                        std::iter::repeat_n(if i % 2 == 0 { start } else { 19 - start }, n + 1)
                    })
                    .collect();
                let gaps = distribute(length - 1 - size, size + 1, rng);
                blocks_row(&word, &gaps, blocks)
            })
            .collect();
    }
    let (first, second) = if blocks == 1 {
        let first = vec![9; size];
        let mut second = first.clone();
        second[rng.below(size as u32) as usize] = 10;
        (first, second)
    } else {
        let mut counts = vec![1; blocks];
        counts[0] = 2;
        counts[1] = 2;
        let extra = distribute(size - counts.iter().sum::<usize>(), blocks, rng);
        let first: Vec<_> = counts
            .iter()
            .zip(&extra)
            .enumerate()
            .flat_map(|(i, (a, b))| std::iter::repeat_n(if i % 2 == 0 { 9 } else { 10 }, a + b))
            .collect();
        let mut second = first.clone();
        let cut = counts[0] + extra[0];
        second.swap(cut - 1, cut);
        (first, second)
    };
    let gaps = distribute(length - 1 - size, size + 1, rng);
    [first, second]
        .iter()
        .map(|w| blocks_row(w, &gaps, blocks))
        .collect()
}
fn recall(s: &Value, rng: &mut Random, length: usize) -> data::Example {
    let symbols = s["symbols"].as_u64().unwrap() as usize;
    let pairs = s["pairs"].as_u64().unwrap() as usize;
    let queries = s["queries"].as_u64().unwrap() as usize;
    let overwrites = s["overwrites"].as_u64().unwrap_or(0) as usize;
    let distinct: Vec<_> = rng
        .sample(symbols, pairs - overwrites)
        .into_iter()
        .map(|k| 36 + k as i64)
        .collect();
    let mut keys = distinct.clone();
    keys.extend((0..overwrites).map(|_| distinct[rng.below(distinct.len() as u32) as usize]));
    if overwrites > 0 {
        rng.shuffle(&mut keys);
    }
    let mut current = BTreeMap::new();
    let mut values = vec![];
    let first_value = 36 + symbols as i64;
    for key in &keys {
        let mut value = first_value
            + rng.below((symbols - usize::from(current.contains_key(key))) as u32) as i64;
        if current.get(key).is_some_and(|&old| value >= old) {
            value += 1;
        }
        current.insert(*key, value);
        values.push(value);
    }
    let mut tokens = vec![1];
    tokens.extend((1..length).map(|_| first_value + rng.below(symbols as u32) as i64));
    for i in 0..pairs {
        tokens[1 + 2 * i] = keys[i];
        tokens[2 + 2 * i] = values[i];
    }
    let end = 1 + 2 * pairs;
    let gap = s["query_gap"].as_u64().unwrap() as usize;
    let alpha = s["alpha"].as_f64().unwrap();
    let mut positions: Vec<_> = (end + gap..length)
        .map(|p| {
            (
                (-(rng.random().max(f64::MIN_POSITIVE)).ln()).ln() + alpha * (p as f64).ln(),
                p,
            )
        })
        .collect();
    positions.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let selected = rng.sample(distinct.len(), queries);
    let mut labels = vec![-100; length];
    for ((_, pos), i) in positions.iter().take(queries).zip(selected) {
        tokens[*pos] = distinct[i];
        labels[*pos] = current[&distinct[i]];
    }
    wire::example("mqar", tokens, labels)
}
fn parity(rng: &mut Random, length: usize) -> data::Example {
    let _hints = wire::format_rng(rng);
    let mut prompt = vec![1];
    let mut odd = false;
    for _ in 0..length {
        let t = 21 + rng.below(2) as i64;
        prompt.push(t);
        odd ^= t == 22;
    }
    prompt.push(18);
    wire::answered("parity", prompt, vec![if odd { 25 } else { 24 }, 17], 2)
}
pub fn key(e: &data::Example) -> &[i64] {
    if e.prompt.is_empty() {
        &e.tokens
    } else {
        &e.prompt
    }
}

/// Bounded source stream, including separated_split's original inspected-row
/// limit. Excluded and duplicate inputs never alter the RNG draw sequence.
pub fn generate(
    s: &Value,
    split: &str,
    count: usize,
    excluded: &BTreeSet<Vec<i64>>,
    disjoint: bool,
) -> Result<Vec<data::Example>> {
    let maximum = number(s, "length")?;
    let minimum = if s["min_length"].is_null() {
        maximum
    } else {
        number(s, "min_length")?
    };
    if !["train", "validation", "test"].contains(&split)
        || count == 0
        || count > 20000
        || minimum == 0
        || minimum > maximum
        || maximum > 4096
        || count * maximum > 2_000_000
    {
        return Err("bounded source split dimensions refused".into());
    }
    let task = s["task"].as_str().ok_or("missing task")?;
    match task {
        "blocks" => {
            let blocks = number(s, "blocks")?;
            let fraction = s["neutral_fraction"].as_f64().ok_or("missing fraction")?;
            if !(1..=16).contains(&blocks)
                || ![0., 0.25].contains(&fraction)
                || !s["max_neutral_gap"].is_null()
                || ((minimum - 1) as f64 * (1. - fraction)).round_ties_even()
                    < (blocks + if blocks > 1 { 2 } else { 1 }) as f64
            {
                return Err("blocks specification refused".into());
            }
        }
        "mqar" => {
            let (symbols, pairs, queries, gap) = (
                number(s, "symbols")?,
                number(s, "pairs")?,
                number(s, "queries")?,
                number(s, "query_gap")?,
            );
            let overwrites = s["overwrites"].as_u64().unwrap_or(0) as usize;
            let alpha = s["alpha"].as_f64().ok_or("missing alpha")?;
            if symbols == 0
                || symbols > 1024
                || pairs == 0
                || overwrites >= pairs
                || pairs - overwrites > symbols
                || (overwrites > 0 && symbols < 2)
                || queries == 0
                || queries > pairs - overwrites
                || gap > 4096
                || minimum < 1 + 2 * pairs + queries + gap
                || !alpha.is_finite()
                || alpha < 0.
            {
                return Err("recall specification refused".into());
            }
        }
        "parity" => {
            if maximum > 1024 || s["scratchpad"] != "none" || s["index_hints"] != false {
                return Err("parity specification refused".into());
            }
        }
        _ => return Err("unsupported Basis family".into()),
    }
    let mut rng = wire::rng(s, split);
    let mut rows = vec![];
    let mut seen = excluded.clone();
    let mut inspected = 0;
    let limit = if disjoint {
        4096.max(64 * count)
    } else {
        count
    };
    while inspected < limit && rows.len() < count {
        let length = minimum + rng.below((maximum - minimum + 1) as u32) as usize;
        let mut pair = match task {
            "blocks" => blocks_pair(s, &mut rng, length),
            "mqar" => (0..2).map(|_| recall(s, &mut rng, length)).collect(),
            _ => (0..2).map(|_| parity(&mut rng, length)).collect(),
        };
        if rng.random() < 0.5 {
            pair.reverse();
        }
        for row in pair {
            if inspected == limit || rows.len() == count {
                break;
            }
            inspected += 1;
            independent::check(s, &row)?;
            if !disjoint || seen.insert(key(&row).to_vec()) {
                rows.push(row);
            }
        }
    }
    if rows.len() != count {
        return Err("source cannot supply the required disjoint inputs".into());
    }
    Ok(rows)
}

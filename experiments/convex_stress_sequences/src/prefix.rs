//! Port of frozen synthetic/prefixes.py. Matched words retain symbol counts
//! and neutral gaps while changing their prefix labels. Random words broaden
//! support. Generator rules never enter the predictor.
use crate::{Random, data::Example, wire};
use serde_json::Value;

fn balanced(size: usize, maximum: i64, rng: &mut Random) -> Vec<i64> {
    let mut balance = 0;
    (0..size)
        .map(|i| {
            let open = balance == 0
                || (balance < maximum && balance < (size - i) as i64 && rng.random() < 0.5);
            balance += if open { 1 } else { -1 };
            if open { 7 } else { 8 }
        })
        .collect()
}

fn labels(task: &str, tokens: Vec<i64>, blocks: usize) -> Example {
    let mut targets = vec![-100];
    let (mut balance, mut invalid, mut first, mut previous, mut runs) = (0, false, 0, 0, 0);
    for &token in &tokens[1..] {
        let label = if task == "dyck" {
            balance += i64::from(token == 7) - i64::from(token == 8);
            invalid |= balance < 0;
            if invalid {
                14
            } else if balance == 0 {
                12
            } else {
                13
            }
        } else {
            if token != 11 {
                if first == 0 {
                    first = token;
                }
                runs += usize::from(previous != token);
                previous = token;
            }
            if first == 9 && runs == blocks { 16 } else { 15 }
        };
        targets.push(label);
    }
    wire::example(task, tokens, targets)
}

pub(crate) fn pair(spec: &Value, rng: &mut Random, length: usize) -> Vec<Example> {
    let task = spec["task"].as_str().unwrap();
    let dyck = task == "dyck";
    let blocks = spec["blocks"].as_u64().unwrap_or(0) as usize;
    let maximum = spec["max_balance"].as_i64().unwrap_or(0);
    let mut size = ((length - 1) as f64 * (1. - spec["neutral_fraction"].as_f64().unwrap()))
        .round_ties_even() as usize;
    if dyck {
        size -= size % 2;
    }
    let random_word = |rng: &mut Random| {
        if dyck {
            let mut balance = 0;
            (0..size)
                .map(|_| {
                    let open = balance == -maximum || (balance < maximum && rng.random() < 0.5);
                    balance += if open { 1 } else { -1 };
                    if open { 7 } else { 8 }
                })
                .collect::<Vec<_>>()
        } else {
            let runs = 1 + rng.below(size.min(2 * blocks + 3) as u32) as usize;
            let counts = wire::distribute(size - runs, runs, rng);
            let start = 9 + rng.below(2) as i64;
            counts
                .into_iter()
                .enumerate()
                .flat_map(|(i, n)| {
                    std::iter::repeat_n(if i % 2 == 0 { start } else { 19 - start }, n + 1)
                })
                .collect()
        }
    };
    if rng.random() < 1. / 3. {
        return (0..2)
            .map(|_| {
                let word = random_word(rng);
                let gaps = wire::distribute(length - 1 - size, size + 1, rng);
                labels(task, wire::gaps(&word, &gaps), blocks)
            })
            .collect();
    }
    let (first, second) = if dyck {
        let cut = 2 * (1 + rng.below(((size - 4) / 2) as u32) as usize);
        let mut first = balanced(cut, maximum, rng);
        first.extend(balanced(size - cut, maximum, rng));
        let candidates: Vec<_> = (cut + 1..size - 1).filter(|&i| first[i] == 8).collect();
        let closing = if maximum == 1 {
            candidates[0]
        } else {
            candidates[rng.below(candidates.len() as u32) as usize]
        };
        let mut second = first.clone();
        second.swap(cut, closing);
        (first, second)
    } else if blocks == 1 {
        let first = vec![9; size];
        let mut second = first.clone();
        second[rng.below(size as u32) as usize] = 10;
        (first, second)
    } else {
        let mut counts = vec![1; blocks];
        counts[0] = 2;
        counts[1] = 2;
        let extra = wire::distribute(size - counts.iter().sum::<usize>(), blocks, rng);
        let first: Vec<_> = counts
            .iter()
            .zip(&extra)
            .enumerate()
            .flat_map(|(i, (&a, &b))| std::iter::repeat_n(if i % 2 == 0 { 9 } else { 10 }, a + b))
            .collect();
        let mut second = first.clone();
        let cut = counts[0] + extra[0];
        second.swap(cut - 1, cut);
        (first, second)
    };
    let gaps = wire::distribute(length - 1 - size, size + 1, rng);
    [first, second]
        .iter()
        .map(|word| labels(task, wire::gaps(word, &gaps), blocks))
        .collect()
}

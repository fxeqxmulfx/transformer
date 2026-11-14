//! Read-only source: synthetic/prefixes.py TypedDyck in task318's snapshot.
//! Preserve every conditional random draw, forced first two types, paired
//! corruption, gap allocation and row reversal. This module only makes data.
use crate::{Random, data::Example, wire};

pub(crate) fn pairs(types: usize) -> Vec<(i64, i64)> {
    std::iter::once((7, 8))
        .chain((0..types - 1).map(|i| (36 + 2 * i as i64, 37 + 2 * i as i64)))
        .collect()
}
fn balanced(size: usize, maximum: i64, rng: &mut Random) -> Vec<bool> {
    let mut balance = 0;
    (0..size)
        .map(|i| {
            let open = balance == 0
                || (balance < maximum && balance < (size - i) as i64 && rng.random() < 0.5);
            balance += if open { 1 } else { -1 };
            open
        })
        .collect()
}
pub(crate) fn labels(tokens: Vec<i64>, types: usize) -> Example {
    let pairs = pairs(types);
    let (mut stack, mut invalid, mut targets) = (vec![], false, vec![-100]);
    for &token in &tokens[1..] {
        if let Some(&(_, closing)) = pairs.iter().find(|&&(opening, _)| opening == token) {
            stack.push(closing);
        } else if pairs.iter().any(|&(_, closing)| closing == token) {
            invalid |= stack.pop() != Some(token);
        } else {
            assert_eq!(token, 11);
        }
        targets.push(if invalid {
            14
        } else if stack.is_empty() {
            12
        } else {
            13
        });
    }
    wire::example("dyck2", tokens, targets)
}
pub(crate) fn matched(
    types: usize,
    size: usize,
    maximum: i64,
    rng: &mut Random,
) -> (Vec<i64>, Vec<i64>) {
    let pairs = pairs(types);
    let skeleton = balanced(size, maximum, rng);
    let (mut word, mut stack, mut opened) = (vec![], vec![], 0);
    for open in skeleton {
        if open {
            let kind = if opened < 2 {
                opened
            } else {
                rng.below(types as u32) as usize
            };
            opened += 1;
            word.push(pairs[kind].0);
            stack.push(pairs[kind].1);
        } else {
            word.push(stack.pop().unwrap());
        }
    }
    let closings: Vec<_> = (0..size)
        .filter(|&i| pairs.iter().any(|&(_, c)| c == word[i]))
        .collect();
    let first = closings[rng.below(closings.len() as u32) as usize];
    let other: Vec<_> = closings
        .into_iter()
        .filter(|&i| word[i] != word[first])
        .collect();
    let second = other[rng.below(other.len() as u32) as usize];
    let mut corrupted = word.clone();
    corrupted.swap(first, second);
    (word, corrupted)
}
pub(crate) fn pair(types: usize, length: usize, maximum: i64, rng: &mut Random) -> Vec<Example> {
    let mut size = ((length - 1) as f64 * 0.75).round_ties_even() as usize;
    size -= size % 2;
    if rng.random() < 1. / 3. {
        return (0..2)
            .map(|_| {
                let pairs = pairs(types);
                let mut balance = 0;
                let word: Vec<_> = (0..size)
                    .map(|_| {
                        let open = balance == -maximum || (balance < maximum && rng.random() < 0.5);
                        let selected = pairs[rng.below(types as u32) as usize];
                        balance += if open { 1 } else { -1 };
                        if open { selected.0 } else { selected.1 }
                    })
                    .collect();
                let gaps = wire::distribute(length - 1 - size, size + 1, rng);
                labels(wire::gaps(&word, &gaps), types)
            })
            .collect();
    }
    let (first, second) = matched(types, size, maximum, rng);
    let gaps = wire::distribute(length - 1 - size, size + 1, rng);
    [first, second]
        .iter()
        .map(|word| labels(wire::gaps(word, &gaps), types))
        .collect()
}

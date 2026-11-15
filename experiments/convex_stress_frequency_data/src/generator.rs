//! Literal data/scorer port of Symbols/Histogram/DoubleHistogram/Mode/MostFrequent
//! in frozen synthetic/sequences.py. No model may import this answer constructor.
use crate::*;

pub(crate) struct Config<'a> {
    pub task: &'a str,
    pub symbols: usize,
    pub bos: bool,
    pub scratchpad: &'a str,
    pub numbers: bool,
}

pub(crate) fn counts(word: &[i64]) -> Vec<(i64, usize)> {
    let mut out: Vec<(i64, usize)> = vec![];
    for &token in word {
        if let Some(pair) = out.iter_mut().find(|p| p.0 == token) {
            pair.1 += 1;
        } else {
            out.push((token, 1));
        }
    }
    out
}

pub(crate) fn sample(config: &Config<'_>, size: usize, rng: &mut Random) -> data::Example {
    // Answered.sample consumes a format seed even though these formats use none
    // of its independent draws. Discarding four limbs preserves the main stream.
    for _ in 0..4 {
        rng.bits(32);
    }
    let cap = if config.task == "mode" { 5 } else { 8 };
    let maximum = config.symbols.min(cap).min(size.max(2));
    let pool_size = 2 + rng.below((maximum - 1) as u32) as usize;
    let pool = rng.sample(config.symbols, pool_size);
    let mut word: Vec<_> = (0..size)
        .map(|_| 36 + pool[rng.below(pool.len() as u32) as usize] as i64)
        .collect();
    if config.task == "mode" {
        repair_tie(&mut word, rng);
    }
    from_word(config, &word)
}

pub(crate) fn repair_tie(word: &mut [i64], rng: &mut Random) {
    let counts = counts(word);
    let highest = counts.iter().map(|p| p.1).max().expect("nonempty word");
    let winners: Vec<_> = counts
        .iter()
        .filter(|p| p.1 == highest)
        .map(|p| p.0)
        .collect();
    if winners.len() > 1 {
        let winner = winners[rng.below(winners.len() as u32) as usize];
        let donors: Vec<_> = winners.into_iter().filter(|&t| t != winner).collect();
        let donor = donors[rng.below(donors.len() as u32) as usize];
        let positions: Vec<_> = word
            .iter()
            .enumerate()
            .filter(|(_, t)| **t == donor)
            .map(|(i, _)| i)
            .collect();
        word[positions[rng.below(positions.len() as u32) as usize]] = winner;
    }
}

pub(crate) fn from_word(config: &Config<'_>, word: &[i64]) -> data::Example {
    let counts = counts(word);
    let number_base = (36 + config.symbols) as i64;
    let mut answer = vec![];
    match config.task {
        "histogram" => {
            if config.bos {
                answer.push(1);
            }
            answer.extend(
                word.iter()
                    .map(|t| number_base + counts.iter().find(|p| p.0 == *t).unwrap().1 as i64),
            );
        }
        "histogram2" => {
            answer.push(1);
            for t in word {
                let frequency = counts.iter().find(|p| p.0 == *t).unwrap().1;
                answer
                    .push(number_base + counts.iter().filter(|p| p.1 == frequency).count() as i64);
            }
        }
        "mode" => {
            let maximum = counts.iter().map(|p| p.1).max().unwrap();
            let winner = counts.iter().find(|p| p.1 == maximum).unwrap().0;
            let mut order: Vec<_> = (0..counts.len()).collect();
            if config.scratchpad == "counts" {
                order.sort_by_key(|&i| (counts[i].1, i));
            }
            if config.numbers {
                for i in order {
                    answer.extend([number_base + counts[i].1 as i64, counts[i].0]);
                }
            }
            answer.push(winner);
        }
        "most_freq" => {
            let mut order: Vec<_> = (0..counts.len()).collect();
            order.sort_by_key(|&i| (std::cmp::Reverse(counts[i].1), i));
            answer.push(1);
            answer.extend(order.into_iter().map(|i| counts[i].0));
            answer.resize(word.len() + 1, 1);
        }
        _ => unreachable!("validated source task"),
    }
    answer.push(17);
    let mut prompt = if config.bos { vec![1] } else { vec![] };
    prompt.extend_from_slice(word);
    prompt.push(18);
    let mut tokens = prompt.clone();
    tokens.extend_from_slice(&answer[..answer.len() - 1]);
    let mut targets = vec![-100; prompt.len() - 1];
    targets.extend_from_slice(&answer);
    let mut row = wire::example(config.task, tokens, targets);
    row.prompt = prompt;
    row.answer = answer;
    row.limit = if config.task == "mode" {
        2 + if config.numbers {
            2 * word.len().min(config.symbols)
        } else {
            0
        }
    } else {
        word.len() + 1 + usize::from(config.bos)
    };
    row
}

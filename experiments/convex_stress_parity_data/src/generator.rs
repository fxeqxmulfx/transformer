//! Source port of synthetic/bits.py and Answered.sample in base.py.
//! Formats share main RNG draws; hints have a separately seeded MT19937 stream.
use crate::*;

pub(crate) fn hint_rng(rng: &mut Random) -> Random {
    // CPython getrandbits(128) writes four little-endian 32-bit limbs.
    // Integer reseeding discards leading zero limbs, retaining one for zero.
    let mut words: Vec<_> = (0..4).map(|_| rng.bits(32)).collect();
    while words.len() > 1 && words.last() == Some(&0) {
        words.pop();
    }
    Random::words(&words)
}

pub(crate) fn sample(
    scratchpad: &str,
    hinted: bool,
    number_base: usize,
    number_limit: usize,
    size: usize,
    rng: &mut Random,
) -> data::Example {
    let mut hints = hint_rng(rng);
    let bits: Vec<i64> = (0..size).map(|_| 21 + rng.below(2) as i64).collect();
    let start = if hinted {
        hints.below((number_limit - size + 1) as u32) as usize
    } else {
        0
    };
    from_bits(scratchpad, hinted.then_some(number_base + start), &bits)
}

pub(crate) fn from_bits(scratchpad: &str, start: Option<usize>, bits: &[i64]) -> data::Example {
    let mut prompt = vec![1];
    let mut answer = if scratchpad == "none" {
        vec![]
    } else {
        vec![24]
    };
    let mut odd = false;
    for (index, &bit) in bits.iter().enumerate() {
        if let Some(start) = start {
            prompt.push((start + index) as i64);
        }
        prompt.push(bit);
        odd ^= bit == 22;
        if scratchpad == "running" || (scratchpad == "ones" && bit == 22) {
            if let Some(start) = start {
                answer.push((start + index) as i64);
            }
            answer.push(if odd { 25 } else { 24 });
        }
    }
    prompt.push(18);
    if scratchpad == "none" {
        answer.push(if odd { 25 } else { 24 });
    }
    answer.push(17);
    let mut tokens = prompt.clone();
    tokens.extend_from_slice(&answer[..answer.len() - 1]);
    let mut targets = vec![-100; prompt.len() - 1];
    targets.extend_from_slice(&answer);
    let mut row = wire::example("parity", tokens, targets);
    row.prompt = prompt;
    row.limit = if scratchpad == "none" {
        2
    } else {
        (if start.is_some() { 2 } else { 1 }) * bits.len() + 2
    };
    row.answer = answer;
    row
}

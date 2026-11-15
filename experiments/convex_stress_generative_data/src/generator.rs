//! Read-only source port: synthetic/{sequences,base,records}.py at task318.
//! Preserve the unused128-bit format draw, pair order and ordinary symbol draws.
use crate::*;

pub(crate) fn sample(
    task: &str,
    symbols: usize,
    unique: bool,
    size: usize,
    rng: &mut Random,
) -> data::Example {
    for _ in 0..4 {
        let _ = rng.bits(32);
    }
    let word: Vec<i64> = if unique {
        rng.sample(symbols, size)
            .into_iter()
            .map(|v| 36 + v as i64)
            .collect()
    } else {
        let limit = if task == "sort" { 8 } else { 2 };
        let upper = symbols.min(limit).min(size.max(2));
        let count = 2 + rng.below((upper - 1) as u32) as usize;
        let pool = rng.sample(symbols, count);
        (0..size)
            .map(|_| 36 + pool[rng.below(count as u32) as usize] as i64)
            .collect()
    };
    from_word(task, &word)
}

pub(crate) fn from_word(task: &str, word: &[i64]) -> data::Example {
    let mut prompt = vec![1];
    prompt.extend_from_slice(word);
    prompt.push(18);
    let mut answer = word.to_vec();
    match task {
        "copy" => {}
        "reverse" => answer.reverse(),
        "sort" => answer.sort_unstable(),
        _ => unreachable!("validated rewrite family"),
    }
    answer.push(17);
    let mut tokens = prompt.clone();
    tokens.extend_from_slice(&answer[..answer.len() - 1]);
    let mut targets = vec![-100; prompt.len() - 1];
    targets.extend_from_slice(&answer);
    let mut row = wire::example(task, tokens, targets);
    row.prompt = prompt;
    row.limit = word.len() + 1;
    row.answer = answer;
    row
}

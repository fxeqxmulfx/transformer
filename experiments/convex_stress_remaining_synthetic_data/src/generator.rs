//! Task sampling only. No source target routine is exposed to a predictor.
use crate::*;
use serde_json::Value;

pub fn generate(s: &Value, split: &str, count: usize) -> Result<Vec<data::Example>> {
    spec::validate(s, count)?;
    if !["train", "validation", "test"].contains(&split) {
        return Err("invalid split".into());
    }
    let task = spec::task(s)?;
    let length = spec::number(s, "length")?;
    let minimum = spec::minimum(s)?;
    let program = if task == "crasp" {
        Some(crasp::select(
            spec::number(s, "formula_depth")?,
            spec::number(s, "formula_seed")? as u32,
        )?)
    } else {
        None
    };
    let mut rng = wire::rng(s, split);
    let mut rows = Vec::with_capacity(count + 1);
    while rows.len() < count {
        let size = minimum + rng.below((length - minimum + 1) as u32) as usize;
        if task == "random_lm" {
            let symbols = spec::symbols(s)?;
            let base = 36 + symbols;
            let prompt = vec![1, (base + size) as i64, 18];
            let mut answer: Vec<_> = (0..size)
                .map(|_| 36 + rng.below(symbols as u32) as i64)
                .collect();
            answer.push(17);
            rows.push(wire::answered(task, prompt, answer, size + 1));
            continue;
        }
        let mut pair = vec![];
        for _ in 0..2 {
            if let Some(selected) = &program {
                let prefix = if rng.random() < 0.5 {
                    selected.witnesses[rng.below(2) as usize].clone()
                } else {
                    vec![]
                };
                let mut word: Vec<_> = prefix.into_iter().take(size - 1).collect();
                word.extend((word.len()..size - 1).map(|_| crasp::SYMBOLS[rng.below(3) as usize]));
                let targets = std::iter::once(-100)
                    .chain(
                        selected
                            .formula
                            .evaluate(&word)?
                            .iter()
                            .map(|&b| if b { 16 } else { 15 }),
                    )
                    .collect();
                pair.push(wire::example(
                    task,
                    std::iter::once(1).chain(word).collect(),
                    targets,
                ));
            } else {
                // All Answered tasks consume128 format bits, even without hints.
                let mut hint_rng = wire::format_rng(&mut rng);
                if task == "count" {
                    let base = 36 + spec::symbols(s)?;
                    let start = 1 + rng.below((spec::number(s, "number_limit")? - size + 1) as u32)
                        as usize;
                    let end = start + size - 1;
                    let prompt = vec![1, (base + start) as i64, (base + end) as i64, 18];
                    let answer = (base + start..=base + end)
                        .map(|x| x as i64)
                        .chain(std::iter::once(17))
                        .collect();
                    pair.push(wire::answered(task, prompt, answer, end - start + 2));
                } else {
                    pair.push(addition::sample(s, size, &mut rng, &mut hint_rng)?);
                }
            }
        }
        if rng.random() < 0.5 {
            pair.reverse();
        }
        rows.extend(pair);
    }
    rows.truncate(count);
    Ok(rows)
}

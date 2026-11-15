//! Frozen synthetic/arithmetic.py: exact sampling and prompt-based sum oracle.
//! Integer and digit-wise arithmetic are checked separately. No learned model
//! imports the target oracle. Bounds keep the integer oracle inside u128.
use crate::*;
use serde_json::Value;

pub fn carry_profile(left: &[i64], right: &[i64]) -> Vec<usize> {
    let mut carry = 0;
    left.iter()
        .rev()
        .zip(right.iter().rev())
        .map(|(a, b)| {
            carry = usize::from(a + b + carry as i64 >= 10);
            carry
        })
        .collect()
}
pub fn longest_carry(left: &[i64], right: &[i64]) -> usize {
    let mut current = 0;
    carry_profile(left, right)
        .into_iter()
        .map(|c| {
            current = if c == 1 { current + 1 } else { 0 };
            current
        })
        .max()
        .unwrap_or(0)
}

pub type Operands = (Vec<i64>, Vec<i64>, Vec<i64>);
pub fn parse(s: &Value, prompt: &[i64]) -> Result<Operands> {
    if prompt.first() != Some(&1)
        || prompt.last() != Some(&18)
        || prompt.iter().filter(|&&t| t == 20).count() != 1
    {
        return Err("addition needs BOS, two operands, PLUS and SEP".into());
    }
    let split = prompt.iter().position(|&t| t == 20).unwrap();
    if split == 0 || split >= prompt.len() - 1 {
        return Err("empty addition operand".into());
    }
    let mut operands = vec![];
    let mut hints = vec![];
    for side in [&prompt[1..split], &prompt[split + 1..prompt.len() - 1]] {
        let digits = if s["index_hints"] == true {
            if !side.len().is_multiple_of(2) {
                return Err("hint must precede every digit".into());
            }
            let indices: Vec<_> = side.iter().step_by(2).copied().collect();
            let base = (36 + spec::symbols(s)?) as i64;
            let limit = spec::number(s, "number_limit")? as i64;
            if indices.is_empty()
                || indices.iter().any(|&x| x < base || x > base + limit)
                || indices.windows(2).any(|w| w[1] != w[0] + 1)
            {
                return Err("invalid addition hint interval".into());
            }
            hints.push(indices);
            side.iter().skip(1).step_by(2).copied().collect::<Vec<_>>()
        } else {
            side.to_vec()
        };
        if !(2..=33).contains(&digits.len())
            || digits[0] != 26
            || digits.iter().any(|d| !(26..36).contains(d))
        {
            return Err("invalid padded decimal operand".into());
        }
        operands.push(digits.iter().map(|x| x - 26).collect::<Vec<_>>());
    }
    if operands[0].len() != operands[1].len() || (!hints.is_empty() && hints[0] != hints[1]) {
        return Err("operand widths or index hints differ".into());
    }
    Ok((
        operands.remove(0),
        operands.remove(0),
        hints.into_iter().next().unwrap_or_default(),
    ))
}
pub fn limit(s: &Value, prompt: &[i64]) -> Result<usize> {
    Ok(parse(s, prompt)?.0.len() * (if s["index_hints"] == true { 2 } else { 1 }) + 1)
}
pub fn solve(s: &Value, prompt: &[i64]) -> Result<Vec<i64>> {
    let (left, right, hints) = parse(s, prompt)?;
    let integer = |digits: &[i64]| -> Result<u128> {
        digits.iter().try_fold(0u128, |a, &d| {
            a.checked_mul(10)
                .and_then(|a| a.checked_add(d as u128))
                .ok_or_else(|| "decimal oracle overflow".into())
        })
    };
    let sum = integer(&left)?
        .checked_add(integer(&right)?)
        .ok_or("sum overflow")?;
    let text = format!("{sum:0width$}", width = left.len());
    if text.len() != left.len() {
        return Err("sum exceeded padded width".into());
    }
    let digits: Vec<_> = text.bytes().map(|b| (b - b'0') as i64).collect();
    let mut order: Vec<_> = (0..digits.len()).collect();
    match s["addition_order"].as_str() {
        Some("forward") => {}
        Some("reverse") => order.reverse(),
        _ => return Err("invalid sum order".into()),
    }
    let mut answer = vec![];
    for index in order {
        if !hints.is_empty() {
            answer.push(hints[index]);
        }
        answer.push(26 + digits[index]);
    }
    answer.push(17);
    Ok(answer)
}
pub(crate) fn sample(
    s: &Value,
    size: usize,
    rng: &mut Random,
    hint_rng: &mut Random,
) -> Result<data::Example> {
    let (left, right) = if s["carry_length"].is_null() && s["carry_sampling"] == "standard" {
        let mut operands = vec![];
        for _ in 0..2 {
            let width = 1 + rng.below(size as u32) as usize;
            let mut digits = vec![0; size + 1 - width];
            digits.push(1 + rng.below(9) as i64);
            digits.extend((1..width).map(|_| rng.below(10) as i64));
            operands.push(digits);
        }
        (operands.remove(0), operands.remove(0))
    } else {
        let length = if s["carry_length"].is_null() {
            rng.below(size as u32 + 1) as usize
        } else {
            spec::number(s, "carry_length")?
        };
        if length > size {
            return Err("carry longer than operand".into());
        }
        let start = rng.below((size - length + 1) as u32) as usize;
        let (mut left, mut right) = (vec![], vec![]);
        for position in 0..size {
            let (low, high) = if length > 0 && position == start {
                (10, 18)
            } else if start < position && position < start + length {
                (9, 9)
            } else {
                (0, 8)
            };
            let pairs: Vec<_> = (0..10)
                .flat_map(|a| (0..10).map(move |b| (a, b)))
                .filter(|(a, b)| low <= a + b && a + b <= high)
                .collect();
            let (a, b) = pairs[rng.below(pairs.len() as u32) as usize];
            left.push(a);
            right.push(b);
        }
        left.push(0);
        right.push(0);
        left.reverse();
        right.reverse();
        (left, right)
    };
    if !s["carry_length"].is_null()
        && longest_carry(&left, &right) != spec::number(s, "carry_length")?
    {
        return Err("sample violated requested carry chain".into());
    }
    let hints = if s["index_hints"] == true {
        let start = hint_rng.below((spec::number(s, "number_limit")? - size + 1) as u32) as usize;
        let base = 36 + spec::symbols(s)?;
        (0..=size)
            .map(|i| (base + start + i) as i64)
            .collect::<Vec<_>>()
    } else {
        vec![]
    };
    let mut prompt = vec![1];
    for (side, digits) in [left, right].iter().enumerate() {
        if side == 1 {
            prompt.push(20);
        }
        for (index, digit) in digits.iter().enumerate() {
            if !hints.is_empty() {
                prompt.push(hints[index]);
            }
            prompt.push(26 + digit);
        }
    }
    prompt.push(18);
    let limit = limit(s, &prompt)?;
    let answer = solve(s, &prompt)?;
    if answer.len() != limit {
        return Err("sum oracle disagrees with input horizon".into());
    }
    Ok(wire::answered("addition", prompt, answer, limit))
}

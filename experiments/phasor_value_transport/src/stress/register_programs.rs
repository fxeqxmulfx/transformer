//! DiffGPT 5a37c515, mqar_div97.programs and _program_attempt.
//! A register write uses the current dividend and divisor of its own split.
use super::{Example, PRIME, Pcg, power, wire};

const LOADS: usize = 8;

struct Frame {
    keys: [usize; LOADS],
    order: [usize; LOADS],
    held: [u32; LOADS],
    used: Vec<(u32, u32)>,
    tokens: Vec<i64>,
    valid: bool,
}

fn attempt(
    rng: &mut Pcg,
    count: usize,
    steps: usize,
    allowed: &[bool],
    inverse: &[u32],
    code: &[i64],
) -> Vec<Example> {
    let mut frames: Vec<_> = (0..count)
        .map(|_| {
            let mut alphabet: [usize; 32] = std::array::from_fn(|i| i + 1);
            rng.shuffle(&mut alphabet);
            Frame {
                keys: alphabet[..LOADS].try_into().unwrap(),
                order: std::array::from_fn(|i| i),
                held: [0; LOADS],
                used: Vec::with_capacity(steps),
                tokens: Vec::with_capacity(3 * (LOADS + steps)),
                valid: true,
            }
        })
        .collect();
    for row in &mut frames {
        rng.shuffle(&mut row.order);
    }
    for time in 0..LOADS {
        let fresh: Vec<_> = (0..count).map(|_| 1 + rng.below(PRIME - 1)).collect();
        for (row, value) in frames.iter_mut().zip(fresh) {
            // The source always draws the alternate overwrite, including
            // on the first write to a register in these no-overwrite cases.
            let _changed = rng.below(PRIME - 2);
            let k = row.order[time];
            row.held[k] = value;
            row.tokens
                .extend([code[96 - row.keys[k]], code[0], code[96 + value as usize]]);
        }
    }
    for _ in 0..steps {
        for row in &mut frames {
            let mut best = None;
            let mut keys = (0, 0);
            for (i, &a) in row.held.iter().enumerate() {
                for (j, &b) in row.held.iter().enumerate() {
                    let draw = rng.next64() >> 11;
                    if i != j
                        && allowed[(a * PRIME + b) as usize]
                        && !row.used.contains(&(a, b))
                        && best.is_none_or(|previous| draw > previous)
                    {
                        keys = (i, j);
                        best = Some(draw);
                    }
                }
            }
            row.valid &= best.is_some();
            let (i, j) = keys;
            let (a, b) = (row.held[i], row.held[j]);
            let y = a * inverse[b as usize] % PRIME;
            row.tokens.extend([
                code[96 - row.keys[i]],
                code[96 - row.keys[j]],
                code[96 + y as usize],
            ]);
            row.used.push((a, b));
            row.held[i] = y;
        }
    }
    frames
        .into_iter()
        .filter(|row| row.valid)
        .map(|row| {
            let mut tokens = row.tokens;
            let mut targets = vec![-100; tokens.len() - 1];
            for step in 0..steps {
                let i = 3 * (LOADS + step) + 1;
                targets[i] = tokens[i + 1];
            }
            tokens.pop();
            let mut example = wire::example("associative", tokens, targets);
            example.changes.fill(false);
            example
        })
        .collect()
}

pub(super) fn examples(
    steps: usize,
    equations: &[[u32; 3]],
    stream: u32,
    code: &[i64],
) -> Vec<Example> {
    let mut rng = Pcg::new(&[1, stream]);
    let mut allowed = vec![false; (PRIME * PRIME) as usize];
    for &[a, b, _] in equations {
        if a != 0 {
            allowed[(a * PRIME + b) as usize] = true;
        }
    }
    let count = allowed.iter().filter(|&&v| v).count().div_ceil(steps);
    let inverse: Vec<_> = (0..PRIME).map(|b| power(b, PRIME - 2, PRIME)).collect();
    let mut result = Vec::with_capacity(count);
    while result.len() < count {
        result.extend(attempt(
            &mut rng,
            count - result.len(),
            steps,
            &allowed,
            &inverse,
            code,
        ));
    }
    result
}

//! Input-only bounded summaries. Roles range over every vocabulary symbol;
//! these functions contain no task name, raw label or generator answer rule.
use crate::*;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Program {
    Signed {
        positive: usize,
        negative: usize,
    },
    Runs {
        first: usize,
        second: usize,
        cap: usize,
    },
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Value {
    pub state: usize,
    pub positions: u64,
}
pub fn catalogue(vocabulary: usize) -> Result<Vec<Program>> {
    if !(2..=1024).contains(&vocabulary) || 9 * vocabulary * (vocabulary - 1) > 16384 {
        return Err("generic latent catalogue exceeds16384 candidates".into());
    }
    let mut out = vec![];
    for first in 0..vocabulary {
        for second in 0..vocabulary {
            if first == second {
                continue;
            }
            out.push(Program::Signed {
                positive: first,
                negative: second,
            });
            for cap in 1..=8 {
                out.push(Program::Runs { first, second, cap });
            }
        }
    }
    Ok(out)
}
impl Program {
    pub fn run(self, tokens: &[usize]) -> Value {
        let state = match self {
            Self::Signed { positive, negative } => {
                let (mut total, mut below) = (0i64, false);
                for &t in tokens {
                    total += i64::from(t == positive) - i64::from(t == negative);
                    below |= total < 0;
                }
                3 * usize::from(below) + (total.signum() + 1) as usize
            }
            Self::Runs { first, second, cap } => {
                let (mut start, mut last, mut count) = (0, 0, 0usize);
                for &t in tokens {
                    let role = if t == first {
                        1
                    } else if t == second {
                        2
                    } else {
                        0
                    };
                    if role != 0 {
                        if start == 0 {
                            start = role;
                        }
                        count = (count + usize::from(role != last)).min(cap + 1);
                        last = role;
                    }
                }
                3 * count + start
            }
        };
        Value {
            state,
            positions: tokens.len() as u64,
        }
    }
    /// Independent scalar specification: construct signed prefix sums or a
    /// filtered word and count its unequal adjacent pairs.
    pub fn interpret(self, tokens: &[usize]) -> usize {
        match self {
            Self::Signed { positive, negative } => {
                let increments: Vec<i64> = tokens
                    .iter()
                    .map(|&t| i64::from(t == positive) - i64::from(t == negative))
                    .collect();
                let total: i64 = increments.iter().sum();
                let below = (1..=increments.len()).any(|n| increments[..n].iter().sum::<i64>() < 0);
                3 * usize::from(below) + (total.signum() + 1) as usize
            }
            Self::Runs { first, second, cap } => {
                let word: Vec<_> = tokens
                    .iter()
                    .copied()
                    .filter(|&t| t == first || t == second)
                    .collect();
                let count = usize::from(!word.is_empty())
                    + word.windows(2).filter(|w| w[0] != w[1]).count();
                let start = word.first().map_or(0, |&t| if t == first { 1 } else { 2 });
                3 * count.min(cap + 1) + start
            }
        }
    }
}

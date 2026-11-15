//! Input-only grammar. Copy walks and run summaries extend the generic CM52/53
//! programs at16aafe00; equality and modular counting are fixed operators.
//! No target label, task name, generator state or selected formula is read.
use crate::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Program {
    Lag(usize),
    Walk {
        hops: usize,
        offset: usize,
        latest: bool,
    },
    RunsEq {
        first: usize,
        second: usize,
        runs: usize,
    },
    Residue {
        symbol: usize,
        modulus: usize,
    },
}
pub fn catalogue(input: &input::Input) -> Result<Vec<Program>> {
    let mut bank: Vec<_> = (0..input.maximum_length.min(64))
        .map(Program::Lag)
        .collect();
    for hops in 1..=2 {
        for offset in 1..=4 {
            for latest in [false, true] {
                bank.push(Program::Walk {
                    hops,
                    offset,
                    latest,
                });
            }
        }
    }
    // The fixed small-alphabet grammar is explicitly bounded. Large alphabets
    // still retain the generic copy bank; no target-dependent token role pruning.
    if input.alphabet.len() <= 8 {
        for &first in &input.alphabet {
            for &second in &input.alphabet {
                if first != second {
                    for runs in 1..=16 {
                        bank.push(Program::RunsEq {
                            first,
                            second,
                            runs,
                        });
                    }
                }
            }
        }
        for &symbol in &input.alphabet {
            for modulus in 2..=4 {
                bank.push(Program::Residue { symbol, modulus });
            }
        }
    }
    if bank.is_empty() || bank.len() > 1024 {
        return Err("fixed input program bank refused".into());
    }
    Ok(bank)
}
impl Program {
    pub fn valid(&self, vocabulary: usize) -> bool {
        match *self {
            Self::Lag(lag) => lag < 64,
            Self::Walk { hops, offset, .. } => (1..=2).contains(&hops) && (1..=4).contains(&offset),
            Self::RunsEq {
                first,
                second,
                runs,
            } => {
                first < vocabulary
                    && second < vocabulary
                    && first != second
                    && (1..=16).contains(&runs)
            }
            Self::Residue { symbol, modulus } => symbol < vocabulary && (2..=4).contains(&modulus),
        }
    }
    pub fn run(&self, tokens: &[i64]) -> Option<Atom> {
        if !self.valid(1024)
            || tokens.is_empty()
            || tokens.len() > 4096
            || tokens.iter().any(|&t| !(0..1024).contains(&t))
        {
            return None;
        }
        match *self {
            Self::Lag(lag) => tokens
                .len()
                .checked_sub(lag + 1)
                .map(|i| Atom::Copy(tokens[i] as usize)),
            Self::Walk {
                hops,
                offset,
                latest,
            } => {
                let mut key = *tokens.last()?;
                for _ in 0..hops {
                    let mut found = None;
                    for i in 0..(tokens.len() - 1).saturating_sub(offset) {
                        if tokens[i] == key {
                            found = Some(tokens[i + offset]);
                            if !latest {
                                break;
                            }
                        }
                    }
                    key = found?;
                }
                Some(Atom::Copy(key as usize))
            }
            Self::RunsEq {
                first,
                second,
                runs,
            } => {
                let (mut start, mut last, mut count) = (None, None, 0);
                for &t in tokens {
                    if t as usize == first || t as usize == second {
                        start.get_or_insert(t as usize);
                        count += usize::from(last != Some(t));
                        last = Some(t);
                    }
                }
                Some(Atom::Quantity(i64::from(
                    start == Some(first) && count == runs,
                )))
            }
            Self::Residue { symbol, modulus } => {
                if tokens[0] != 1 || !(2..=4).contains(&modulus) {
                    return None;
                }
                let sep = tokens.iter().position(|&t| t == 18)?;
                if tokens[sep + 1..].contains(&18) {
                    return None;
                }
                match tokens.len() - sep - 1 {
                    0 => Some(Atom::Quantity(
                        (tokens[1..sep]
                            .iter()
                            .filter(|&&t| t as usize == symbol)
                            .count()
                            % modulus) as i64,
                    )),
                    1 => Some(Atom::Role(0)),
                    _ => None,
                }
            }
        }
    }
    /// Separate list/window interpretation, checked on every retained training
    /// prefix and on every scored query. No primary evaluator is called.
    pub fn interpret(&self, tokens: &[i64]) -> Option<Atom> {
        if !self.valid(1024)
            || tokens.is_empty()
            || tokens.len() > 4096
            || tokens.iter().any(|&t| !(0..1024).contains(&t))
        {
            return None;
        }
        match *self {
            Self::Lag(lag) => tokens
                .iter()
                .rev()
                .nth(lag)
                .map(|&t| Atom::Copy(t as usize)),
            Self::Walk {
                hops,
                offset,
                latest,
            } => {
                let mut key = *tokens.last()?;
                for _ in 0..hops {
                    let matches: Vec<_> = tokens[..tokens.len() - 1]
                        .windows(offset + 1)
                        .filter(|w| w[0] == key)
                        .map(|w| w[offset])
                        .collect();
                    key = *(if latest {
                        matches.last()
                    } else {
                        matches.first()
                    })?;
                }
                Some(Atom::Copy(key as usize))
            }
            Self::RunsEq {
                first,
                second,
                runs,
            } => {
                let word: Vec<_> = tokens
                    .iter()
                    .copied()
                    .filter(|&t| t as usize == first || t as usize == second)
                    .collect();
                let count = usize::from(!word.is_empty())
                    + word.windows(2).filter(|w| w[0] != w[1]).count();
                Some(Atom::Quantity(i64::from(
                    word.first().copied() == Some(first as i64) && count == runs,
                )))
            }
            Self::Residue { symbol, modulus } => {
                if tokens.first() != Some(&1) || !(2..=4).contains(&modulus) {
                    return None;
                }
                let parts: Vec<_> = tokens.split(|&t| t == 18).collect();
                if parts.len() != 2 || parts[0].is_empty() {
                    return None;
                }
                match parts[1].len() {
                    0 => {
                        let sum: usize = parts[0][1..]
                            .iter()
                            .map(|&t| usize::from(t as usize == symbol))
                            .sum();
                        Some(Atom::Quantity((sum % modulus) as i64))
                    }
                    1 => Some(Atom::Role(0)),
                    _ => None,
                }
            }
        }
    }
}

//! Fixed176-program causal prior. The BOS/SEP input codec is public; no output
//! label, numeric formula or task identity occurs here. Hash collisions remain
//! possible and cannot be interpreted as target realizability certificates.
use crate::{Result, Work};
use serde::{Deserialize, Serialize};
pub const STATES: usize = 512;
pub const PROGRAMS: usize = 176;
pub const MAX_PREFIX: usize = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Relation {
    Less,
    Equal,
    Compare,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Program {
    Chunk {
        count: usize,
        interleaved: bool,
        context: Option<usize>,
    },
    Lag(usize),
    Walk {
        hops: usize,
        offset: usize,
        latest: bool,
    },
    Relation {
        last_two: bool,
        relation: Relation,
    },
    Length,
    Constant,
}
pub fn catalog() -> Vec<Program> {
    let mut out = Vec::with_capacity(PROGRAMS);
    for count in 1..=8 {
        for interleaved in [false, true] {
            for context in std::iter::once(None).chain((0..8).map(Some)) {
                out.push(Program::Chunk {
                    count,
                    interleaved,
                    context,
                });
            }
        }
    }
    out.extend((0..8).map(Program::Lag));
    for hops in 1..=2 {
        for offset in 1..=4 {
            for latest in [false, true] {
                out.push(Program::Walk {
                    hops,
                    offset,
                    latest,
                });
            }
        }
    }
    for last_two in [false, true] {
        for relation in [Relation::Less, Relation::Equal, Relation::Compare] {
            out.push(Program::Relation { last_two, relation });
        }
    }
    out.extend([Program::Length, Program::Constant]);
    assert_eq!(out.len(), PROGRAMS);
    out
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Alphabet {
    pub vocabulary: usize,
    pub symbols: Vec<i64>,
}
impl Alphabet {
    pub fn validate(&self) -> Result<()> {
        if !(2..=1024).contains(&self.vocabulary)
            || self.symbols.is_empty()
            || self
                .symbols
                .iter()
                .any(|&x| x < 0 || x as usize >= self.vocabulary)
            || self.symbols.windows(2).any(|w| w[0] >= w[1])
        {
            return Err("invalid training-input alphabet".into());
        }
        Ok(())
    }
    pub fn radix(&self) -> usize {
        if self.symbols.len() <= 3 {
            self.symbols.len() + 1
        } else {
            self.vocabulary + 1
        }
    }
    pub fn digit(&self, token: i64) -> usize {
        if self.symbols.len() <= 3 {
            self.symbols
                .binary_search(&token)
                .unwrap_or(self.symbols.len())
        } else {
            token as usize
        }
    }
}
pub fn separator(prefix: &[i64]) -> Option<usize> {
    (prefix.first() == Some(&1))
        .then(|| prefix.iter().position(|&x| x == 18))
        .flatten()
}
pub fn validate_prefix(prefix: &[i64], vocabulary: usize) -> Result<()> {
    if prefix.is_empty()
        || prefix.len() > MAX_PREFIX
        || prefix.iter().any(|&x| x < 0 || x as usize >= vocabulary)
    {
        return Err("invalid visible causal prefix".into());
    }
    Ok(())
}
fn token_state(token: i64) -> usize {
    2 + token as usize % 509
}
pub fn state(
    program: Program,
    alphabet: &Alphabet,
    prefix: &[i64],
    work: &mut Work,
) -> Result<usize> {
    work.charge(32 + 16 * prefix.len() as u128)?;
    let out = match program {
        Program::Chunk {
            count,
            interleaved,
            context,
        } => {
            let Some(sep) = separator(prefix) else {
                return Ok(0);
            };
            let input = &prefix[1..sep];
            if count == 0 || count > 8 || input.is_empty() || !input.len().is_multiple_of(count) {
                return Ok(0);
            }
            let width = input.len() / count;
            let rank = prefix.len() - sep - 1;
            if rank >= width {
                return Ok(1);
            }
            let mut code = 0;
            for block in 0..count {
                let at = if interleaved {
                    rank * count + block
                } else {
                    block * width + rank
                };
                code = (code * alphabet.radix() + alphabet.digit(input[at])) % 509;
            }
            if let Some(at) = context {
                let Some(&token) = input.get(at) else {
                    return Ok(0);
                };
                code = (code * alphabet.radix() + alphabet.digit(token)) % 509;
            }
            2 + code
        }
        Program::Lag(lag) => {
            if lag >= 8 {
                return Ok(0);
            }
            prefix
                .len()
                .checked_sub(lag + 1)
                .map_or(0, |at| token_state(prefix[at]))
        }
        Program::Walk {
            hops,
            offset,
            latest,
        } => {
            if !(1..=2).contains(&hops) || !(1..=4).contains(&offset) {
                return Ok(0);
            }
            let Some(&last) = prefix.last() else {
                return Ok(0);
            };
            let mut key = last;
            for _ in 0..hops {
                let mut found = None;
                for at in 0..prefix.len().saturating_sub(offset + 1) {
                    if prefix[at] == key {
                        found = Some(prefix[at + offset]);
                        if !latest {
                            break;
                        }
                    }
                }
                let Some(value) = found else { return Ok(0) };
                key = value;
            }
            token_state(key)
        }
        Program::Relation { last_two, relation } => {
            if prefix.len() < 2 {
                return Ok(0);
            }
            let a = prefix[if last_two { prefix.len() - 2 } else { 0 }];
            let b = prefix[prefix.len() - 1];
            2 + match relation {
                Relation::Less => usize::from(a < b),
                Relation::Equal => usize::from(a == b),
                Relation::Compare => match a.cmp(&b) {
                    std::cmp::Ordering::Less => 0,
                    std::cmp::Ordering::Equal => 1,
                    std::cmp::Ordering::Greater => 2,
                },
            }
        }
        Program::Length => 2 + prefix.len() % 509,
        Program::Constant => 2,
    };
    Ok(out)
}

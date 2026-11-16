//! Independent causal interpretation used before any fitting. Deliberately
//! materialize a bounded tuple and scan successor candidates in reverse order.
//! Do not call the primary interpreter or its separator/digit helpers.
use crate::{
    Result, Work,
    features::{Alphabet, Program, Relation},
};
pub fn state(
    program: Program,
    alphabet: &Alphabet,
    tokens: &[i64],
    work: &mut Work,
) -> Result<usize> {
    work.charge(48 + 24 * tokens.len() as u128)?;
    match program {
        Program::Chunk {
            count,
            interleaved,
            context,
        } => {
            if tokens.first() != Some(&1) {
                return Ok(0);
            }
            let mut delimiter = None;
            for (i, &x) in tokens.iter().enumerate() {
                if x == 18 {
                    delimiter = Some(i);
                    break;
                }
            }
            let Some(sep) = delimiter else { return Ok(0) };
            let n = sep - 1;
            if count == 0 || count > 8 || n == 0 || !n.is_multiple_of(count) {
                return Ok(0);
            }
            let width = n / count;
            let position = tokens.len() - (sep + 1);
            if position >= width {
                return Ok(1);
            }
            let mut tuple = [0i64; 9];
            let mut length = 0;
            for i in 0..n {
                let selected = if interleaved {
                    i / count == position
                } else {
                    i % width == position
                };
                if selected {
                    tuple[length] = tokens[i + 1];
                    length += 1;
                }
            }
            if let Some(c) = context {
                if c >= n {
                    return Ok(0);
                }
                tuple[length] = tokens[c + 1];
                length += 1;
            }
            let radix = if alphabet.symbols.len() < 4 {
                alphabet.symbols.len() + 1
            } else {
                alphabet.vocabulary + 1
            };
            let mut result = 0;
            for &token in &tuple[..length] {
                let digit = if alphabet.symbols.len() < 4 {
                    alphabet
                        .symbols
                        .iter()
                        .position(|&x| x == token)
                        .unwrap_or(alphabet.symbols.len())
                } else {
                    token as usize
                };
                result = (radix * result + digit) % 509;
            }
            Ok(result + 2)
        }
        Program::Lag(lag) => {
            if lag >= 8 {
                return Ok(0);
            }
            Ok(tokens
                .iter()
                .rev()
                .nth(lag)
                .map_or(0, |&x| 2 + x as usize % 509))
        }
        Program::Walk {
            hops,
            offset,
            latest,
        } => {
            if hops == 0 || hops > 2 || offset == 0 || offset > 4 {
                return Ok(0);
            }
            let Some(&mut_key) = tokens.last() else {
                return Ok(0);
            };
            let mut key = mut_key;
            for _ in 0..hops {
                let mut next = None;
                for (i, &value) in tokens.iter().enumerate().rev() {
                    if i + offset < tokens.len() - 1 && value == key {
                        next = Some(tokens[i + offset]);
                        if latest {
                            break;
                        }
                    }
                }
                match next {
                    Some(x) => key = x,
                    None => return Ok(0),
                }
            }
            Ok(2 + key as usize % 509)
        }
        Program::Relation { last_two, relation } => {
            let mut iter = tokens.iter().rev();
            let (Some(&b), Some(&previous)) = (iter.next(), iter.next()) else {
                return Ok(0);
            };
            let a = if last_two { previous } else { tokens[0] };
            Ok(2 + match relation {
                Relation::Less => {
                    if a < b {
                        1
                    } else {
                        0
                    }
                }
                Relation::Equal => {
                    if a == b {
                        1
                    } else {
                        0
                    }
                }
                Relation::Compare => {
                    if a < b {
                        0
                    } else if a > b {
                        2
                    } else {
                        1
                    }
                }
            })
        }
        Program::Length => Ok(tokens.iter().fold(0, |n, _| (n + 1) % 509) + 2),
        Program::Constant => Ok(2),
    }
}

//! Generic finite range and radix operators. Input token formats are supplied
//! priors (BOS1, separator18, operand delimiter20, decimal digits26..35,
//! indexed numbers100..228); no output role label is supplied here.
use crate::{prefix::Formula, *};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Operator {
    Left,
    Right,
    Sum,
    Distance,
    Product,
    Minimum,
    Maximum,
    Quotient,
    Remainder,
    BitAnd,
    BitOr,
    BitXor,
}
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Emission {
    Digits,
    HintDigit,
    DigitHint,
    Hints,
}
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Program {
    Range {
        reverse: bool,
        inclusive: bool,
    },
    Cursor {
        reverse: bool,
        offset: i64,
    },
    Radix {
        operator: Operator,
        reverse: bool,
        width_offset: i8,
        emission: Emission,
    },
    Prefix(Formula),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Atom {
    Quantity(i64),
    Copy(usize),
    Role(usize),
}

impl Atom {
    pub fn features(self, coordinates: &[f64]) -> Result<Features> {
        Ok(match self {
            Self::Quantity(q) if q.unsigned_abs() <= 524288 => [q as f64 / 128., 1., 0., 0., 0.],
            Self::Copy(c) if c < coordinates.len() => [0., 0., 0., 0., coordinates[c]],
            Self::Role(0) => [0., 0., 1., 0., 0.],
            Self::Role(1) => [0., 0., 0., 1., 0.],
            _ => return Err("latent value outside fixed feature vocabulary".into()),
        })
    }
}

fn operand(tokens: &[usize]) -> Option<(u128, Vec<usize>, usize)> {
    if tokens.is_empty() || tokens.len() > 66 {
        return None;
    }
    let hinted = tokens[0] >= 100;
    let step = if hinted { 2 } else { 1 };
    if !tokens.len().is_multiple_of(step) {
        return None;
    }
    let mut value = 0u128;
    let mut hints = vec![];
    for chunk in tokens.chunks(step) {
        let digit = *chunk.last()?;
        if !(26..=35).contains(&digit) {
            return None;
        }
        if hinted {
            if !(100..=228).contains(&chunk[0]) {
                return None;
            }
            hints.push(chunk[0]);
        }
        value = value.checked_mul(10)?.checked_add((digit - 26) as u128)?;
    }
    Some((value, hints, tokens.len() / step))
}

impl Operator {
    fn apply(&self, a: u128, b: u128) -> Option<u128> {
        Some(match self {
            Self::Left => a,
            Self::Right => b,
            Self::Sum => a.checked_add(b)?,
            Self::Distance => a.abs_diff(b),
            Self::Product => a.checked_mul(b)?,
            Self::Minimum => a.min(b),
            Self::Maximum => a.max(b),
            Self::Quotient => a.checked_div(b)?,
            Self::Remainder => a.checked_rem(b)?,
            Self::BitAnd => a & b,
            Self::BitOr => a | b,
            Self::BitXor => a ^ b,
        })
    }
}

impl Program {
    pub fn run(&self, tokens: &[usize]) -> Option<Atom> {
        if tokens.is_empty() || tokens.len() > 4096 || tokens.iter().any(|&t| t >= 1024) {
            return None;
        }
        if let Self::Prefix(f) = self {
            return f
                .evaluate(crate::prefix::payload(tokens)?)
                .ok()?
                .last()
                .map(|&b| Atom::Quantity(i64::from(b)));
        }
        if tokens[0] != 1 {
            return None;
        }
        let end = tokens.iter().position(|&t| t == 18)?;
        let input = &tokens[1..end];
        let rank = tokens.len() - end - 1;
        match self {
            Self::Range { reverse, inclusive } => {
                if input.len() != 2 || input.iter().any(|&t| !(100..=228).contains(&t)) {
                    return None;
                }
                let (a, b) = (input[0] as i64 - 100, input[1] as i64 - 100);
                let length = a.abs_diff(b) as usize + usize::from(*inclusive);
                if rank >= length {
                    return Some(Atom::Role(0));
                }
                let i = if *reverse { length - 1 - rank } else { rank } as i64;
                Some(Atom::Quantity(a + if b >= a { i } else { -i }))
            }
            Self::Cursor { reverse, offset } => {
                if input.len() != 1 || !(100..=228).contains(&input[0]) || offset.unsigned_abs() > 2
                {
                    return None;
                }
                let length = input[0] - 100;
                if rank >= length {
                    return Some(Atom::Role(0));
                }
                let i = if *reverse { length - 1 - rank } else { rank } as i64;
                Some(Atom::Quantity(i + offset))
            }
            Self::Radix {
                operator,
                reverse,
                width_offset,
                emission,
            } => {
                if !(-1..=1).contains(width_offset) {
                    return None;
                }
                let split = input.iter().position(|&t| t == 20)?;
                let (a, ha, wa) = operand(&input[..split])?;
                let (b, hb, wb) = operand(&input[split + 1..])?;
                if wa != wb || ha != hb {
                    return None;
                }
                let width = wa.checked_add_signed(*width_offset as isize)?;
                if !(1..=33).contains(&width) {
                    return None;
                }
                // A formatter is either defined for the entire output or
                // refused, including digit positions and the terminal role.
                if *emission != Emission::Digits && ha.len() < width {
                    return None;
                }
                let value = operator.apply(a, b)?;
                if value >= 10u128.checked_pow(width as u32)? {
                    return None;
                }
                let pair = matches!(emission, Emission::HintDigit | Emission::DigitHint);
                let stride = if pair { 2 } else { 1 };
                if rank >= width * stride {
                    return Some(Atom::Role(0));
                }
                let position = rank / stride;
                let digit_index = if *reverse {
                    width - 1 - position
                } else {
                    position
                };
                let emit_hint = match emission {
                    Emission::Digits => false,
                    Emission::Hints => true,
                    Emission::HintDigit => rank.is_multiple_of(2),
                    Emission::DigitHint => !rank.is_multiple_of(2),
                };
                if emit_hint {
                    return ha.get(digit_index).copied().map(Atom::Copy);
                }
                let divisor = 10u128.checked_pow((width - 1 - digit_index) as u32)?;
                Some(Atom::Quantity(((value / divisor) % 10) as i64))
            }
            Self::Prefix(_) => unreachable!(),
        }
    }
}

/// Complete finite range/radix bank. Its ordering and membership do not read
/// task identities, labels or heldout examples. Prefix grammar is separate.
pub fn numeric_bank() -> Vec<Program> {
    let mut out = vec![];
    for reverse in [false, true] {
        for inclusive in [false, true] {
            out.push(Program::Range { reverse, inclusive });
        }
        for offset in -2..=2 {
            out.push(Program::Cursor { reverse, offset });
        }
        for operator in [
            Operator::Left,
            Operator::Right,
            Operator::Sum,
            Operator::Distance,
            Operator::Product,
            Operator::Minimum,
            Operator::Maximum,
            Operator::Quotient,
            Operator::Remainder,
            Operator::BitAnd,
            Operator::BitOr,
            Operator::BitXor,
        ] {
            for width_offset in -1..=1 {
                for emission in [
                    Emission::Digits,
                    Emission::HintDigit,
                    Emission::DigitHint,
                    Emission::Hints,
                ] {
                    out.push(Program::Radix {
                        operator: operator.clone(),
                        reverse,
                        width_offset,
                        emission,
                    });
                }
            }
        }
    }
    out
}

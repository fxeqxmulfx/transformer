//! Separate full-sequence rendering and postorder prefix evaluation for audit.
//! This module never imports the data generator or its answer/formula code.
use crate::{
    prefix::{Formula, Term},
    program::{Atom, Emission, Operator, Program},
};

enum Node {
    Symbol(usize),
    Constant(i32),
    Count(usize),
    Neg(usize),
    Add(usize, usize),
    Less(usize, usize),
    Not(usize),
    And(usize, usize),
}
fn term(t: &Term, n: &mut Vec<Node>) -> usize {
    let x = match t {
        Term::Constant(v) => Node::Constant(*v),
        Term::Count(f) => Node::Count(formula(f, n)),
        Term::Negate(t) => Node::Neg(term(t, n)),
        Term::Add(a, b) => Node::Add(term(a, n), term(b, n)),
    };
    n.push(x);
    n.len() - 1
}
fn formula(f: &Formula, n: &mut Vec<Node>) -> usize {
    let x = match f {
        Formula::Symbol(s) => Node::Symbol(*s),
        Formula::Less(a, b) => Node::Less(term(a, n), term(b, n)),
        Formula::Not(f) => Node::Not(formula(f, n)),
        Formula::And(a, b) => Node::And(formula(a, n), formula(b, n)),
    };
    n.push(x);
    n.len() - 1
}
fn prefix(f: &Formula, word: &[usize]) -> Option<Atom> {
    if word.iter().any(|x| ![9, 10, 23].contains(x)) || f.nodes() > 127 {
        return None;
    }
    let mut nodes = vec![];
    let root = formula(f, &mut nodes);
    let mut values = vec![0i32; nodes.len()];
    let mut counts = vec![0i32; nodes.len()];
    for &symbol in word {
        for (i, node) in nodes.iter().enumerate() {
            values[i] = match *node {
                Node::Symbol(s) => i32::from(s == symbol),
                Node::Constant(c) => c,
                Node::Count(j) => {
                    counts[i] = counts[i].checked_add(values[j])?;
                    counts[i]
                }
                Node::Neg(j) => values[j].checked_neg()?,
                Node::Add(a, b) => values[a].checked_add(values[b])?,
                Node::Less(a, b) => i32::from(values[a] < values[b]),
                Node::Not(j) => i32::from(values[j] == 0),
                Node::And(a, b) => i32::from(values[a] != 0 && values[b] != 0),
            };
        }
    }
    Some(Atom::Quantity(i64::from(values[root])))
}
fn operand(tokens: &[usize]) -> Option<(Vec<u8>, Vec<usize>)> {
    if tokens.is_empty() || tokens.len() > 66 {
        return None;
    }
    let hinted = tokens.first()? >= &100;
    let width = if hinted { 2 } else { 1 };
    if !tokens.len().is_multiple_of(width) {
        return None;
    }
    let mut digits = vec![];
    let mut hints = vec![];
    for i in (0..tokens.len()).step_by(width) {
        if hinted {
            if !(100..=228).contains(&tokens[i]) {
                return None;
            }
            hints.push(tokens[i]);
        }
        let d = tokens[i + width - 1].checked_sub(26)?;
        if d > 9 {
            return None;
        }
        digits.push(d as u8);
    }
    Some((digits, hints))
}
fn value(digits: &[u8]) -> Option<u128> {
    digits
        .iter()
        .map(|d| char::from(b'0' + d))
        .collect::<String>()
        .parse()
        .ok()
}
fn sum(a: &[u8], b: &[u8]) -> Vec<u8> {
    let mut reversed = vec![];
    let mut carry = 0;
    for (&a, &b) in a.iter().rev().zip(b.iter().rev()) {
        let s = a + b + carry;
        reversed.push(s % 10);
        carry = s / 10;
    }
    if carry > 0 {
        reversed.push(carry);
    }
    reversed.reverse();
    reversed
}
impl Program {
    pub fn interpret(&self, tokens: &[usize]) -> Option<Atom> {
        if tokens.is_empty() || tokens.len() > 4096 || tokens.iter().any(|&t| t >= 1024) {
            return None;
        }
        if let Self::Prefix(f) = self {
            let word = if tokens[0] == 1 { &tokens[1..] } else { tokens };
            return if word.is_empty() {
                None
            } else {
                prefix(f, word)
            };
        }
        if tokens[0] != 1 {
            return None;
        }
        let stop = tokens.iter().position(|&t| t == 18)?;
        let prompt = &tokens[1..stop];
        let rank = tokens.len() - stop - 1;
        let mut outputs = vec![];
        match self {
            Self::Range { reverse, inclusive } => {
                if prompt.len() != 2 || prompt.iter().any(|x| !(100..=228).contains(x)) {
                    return None;
                }
                let (mut a, b) = (prompt[0] as i64 - 100, prompt[1] as i64 - 100);
                let step = if b >= a { 1 } else { -1 };
                while a != b {
                    outputs.push(Atom::Quantity(a));
                    a += step;
                }
                if *inclusive {
                    outputs.push(Atom::Quantity(b));
                }
                if *reverse {
                    outputs.reverse();
                }
            }
            Self::Cursor { reverse, offset } => {
                if prompt.len() != 1
                    || !(100..=228).contains(&prompt[0])
                    || offset.unsigned_abs() > 2
                {
                    return None;
                }
                outputs = (0..prompt[0] - 100)
                    .map(|i| Atom::Quantity(i as i64 + offset))
                    .collect();
                if *reverse {
                    outputs.reverse();
                }
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
                let middle = prompt.iter().position(|&t| t == 20)?;
                let (a, ha) = operand(&prompt[..middle])?;
                let (b, hb) = operand(&prompt[middle + 1..])?;
                if a.len() != b.len() || ha != hb {
                    return None;
                }
                let width = a.len().checked_add_signed(*width_offset as isize)?;
                if !(1..=33).contains(&width) {
                    return None;
                }
                let mut digits = if *operator == Operator::Sum {
                    sum(&a, &b)
                } else {
                    let (a, b) = (value(&a)?, value(&b)?);
                    let number = match operator {
                        Operator::Left => a,
                        Operator::Right => b,
                        Operator::Distance => a.abs_diff(b),
                        Operator::Product => a.checked_mul(b)?,
                        Operator::Minimum => a.min(b),
                        Operator::Maximum => a.max(b),
                        Operator::Quotient => a.checked_div(b)?,
                        Operator::Remainder => a.checked_rem(b)?,
                        Operator::BitAnd => a & b,
                        Operator::BitOr => a | b,
                        Operator::BitXor => a ^ b,
                        Operator::Sum => unreachable!(),
                    };
                    number.to_string().bytes().map(|b| b - b'0').collect()
                };
                while digits.len() > 1 && digits[0] == 0 {
                    digits.remove(0);
                }
                if digits.len() > width {
                    return None;
                }
                while digits.len() < width {
                    digits.insert(0, 0);
                }
                let indices: Vec<_> = if *reverse {
                    (0..width).rev().collect()
                } else {
                    (0..width).collect()
                };
                for i in indices {
                    let digit = Atom::Quantity(i64::from(digits[i]));
                    match emission {
                        Emission::Digits => outputs.push(digit),
                        Emission::Hints => outputs.push(Atom::Copy(*ha.get(i)?)),
                        Emission::HintDigit => {
                            outputs.push(Atom::Copy(*ha.get(i)?));
                            outputs.push(digit);
                        }
                        Emission::DigitHint => {
                            outputs.push(digit);
                            outputs.push(Atom::Copy(*ha.get(i)?));
                        }
                    }
                }
            }
            Self::Prefix(_) => unreachable!(),
        }
        outputs.get(rank).copied().or(Some(Atom::Role(0)))
    }
}

//! An independent postorder machine. Every count recounts all inclusive past
//! truth values, instead of using the producer's recursive vector prefix sum.
use super::*;
enum Op {
    Sym(i64),
    Const(i64),
    Count(usize),
    Neg(usize),
    Add(usize, usize),
    Lt(usize, usize),
    Not(usize),
    And(usize, usize),
}
fn term(t: &Term, ops: &mut Vec<Op>) -> usize {
    let op = match t {
        Term::Const(n) => Op::Const(*n),
        Term::Count(f) => Op::Count(formula(f, ops)),
        Term::Neg(t) => Op::Neg(term(t, ops)),
        Term::Add(a, b) => Op::Add(term(a, ops), term(b, ops)),
    };
    ops.push(op);
    ops.len() - 1
}
fn formula(f: &Formula, ops: &mut Vec<Op>) -> usize {
    let op = match f {
        Formula::Sym(s) => Op::Sym(*s),
        Formula::Lt(a, b) => Op::Lt(term(a, ops), term(b, ops)),
        Formula::Not(f) => Op::Not(formula(f, ops)),
        Formula::And(a, b) => Op::And(formula(a, ops), formula(b, ops)),
    };
    ops.push(op);
    ops.len() - 1
}
pub(super) fn evaluate(f: &Formula, word: &[i64]) -> Result<Vec<bool>> {
    let work = f.nodes() as u128 * (word.len() as u128 + 1).pow(2);
    if work > 1_000_000_000 {
        return Err("independent formula recount work refused".into());
    }
    let mut ops = vec![];
    let output = formula(f, &mut ops);
    let mut values = vec![vec![0i64; ops.len()]; word.len()];
    for (i, &symbol) in word.iter().enumerate() {
        for (j, op) in ops.iter().enumerate() {
            values[i][j] = match *op {
                Op::Sym(s) => {
                    if !SYMBOLS.contains(&s) {
                        return Err("invalid independent symbol".into());
                    }
                    i64::from(symbol == s)
                }
                Op::Const(n) => n,
                Op::Count(a) => values[..=i].iter().map(|row| row[a]).sum(),
                Op::Neg(a) => values[i][a]
                    .checked_neg()
                    .ok_or("independent negation overflow")?,
                Op::Add(a, b) => values[i][a]
                    .checked_add(values[i][b])
                    .ok_or("independent term overflow")?,
                Op::Lt(a, b) => i64::from(values[i][a] < values[i][b]),
                Op::Not(a) => i64::from(values[i][a] == 0),
                Op::And(a, b) => i64::from(values[i][a] != 0 && values[i][b] != 0),
            };
        }
    }
    Ok(values.iter().map(|row| row[output] != 0).collect())
}

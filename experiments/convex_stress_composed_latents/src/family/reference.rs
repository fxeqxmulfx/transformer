//! Independent compositions: direct-power leaf interpreter, unsigned sums,
//! local residues and bounded subtraction instead of the primary divisions.
use super::*;

fn residue(input: Input, a: usize, b: usize, q: usize) -> usize {
    let (x, y) = (a % q, b % q);
    match input {
        Input::A => x,
        Input::B => y,
        Input::Sum => (x + y) % q,
        Input::Product => (0..x).fold(0, |n, _| (n + y) % q),
        Input::Distance => {
            if a >= b {
                (x + q - y) % q
            } else {
                (y + q - x) % q
            }
        }
    }
}
fn quotient(mut value: usize, divisor: usize) -> usize {
    let mut q = 0;
    while value >= divisor {
        value -= divisor;
        q += 1;
    }
    q
}
pub(super) fn evaluate(evaluator: &Evaluator, program: Program, a: usize, b: usize) -> usize {
    let p = evaluator.vocabulary();
    match program {
        Program::Old(tree) => evaluator.old.reference(tree, a, b),
        Program::FourTerms {
            terms,
            coefficients,
        } => {
            let mut result = 0;
            for (index, [i, j]) in terms.into_iter().enumerate() {
                let value = evaluator
                    .pair
                    .reference(Pair::Monomial { a: i, b: j }, a, b);
                let c = if index == 0 {
                    1
                } else {
                    coefficients[index - 1]
                };
                for _ in 0..c.unsigned_abs() {
                    result = if c < 0 {
                        (result + p - value) % p
                    } else {
                        (result + value) % p
                    };
                }
            }
            result
        }
        Program::Branch {
            control,
            modulus,
            threshold,
            low,
            high,
        } => {
            let leaf = if residue(control, a, b, usize::from(modulus)) < usize::from(threshold) {
                low
            } else {
                high
            };
            (i64::from(leaf.sign) * evaluator.pair.reference(leaf.source, a, b) as i64)
                .rem_euclid(p as i64) as usize
        }
        Program::Recombine {
            source,
            control,
            modulus,
        } => {
            let q = usize::from(modulus);
            let mut base = 0;
            for _ in 0..quotient(evaluator.pair.reference(source, a, b), q) {
                base += q;
            }
            (base + residue(control, a, b, q)) % p
        }
        Program::Digit { source, high } => {
            let value = match source {
                Input::A => a,
                Input::B => b,
                Input::Sum => b + a,
                Input::Product => (0..a).fold(0, |sum, _| sum + b),
                Input::Distance => a.max(b) - a.min(b),
            };
            (if high { quotient(value, p) } else { value }) % p
        }
    }
}

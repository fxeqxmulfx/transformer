//! Independent direct interpreter for the fixed grammar. This interpreter
//! receives no labels and is used on every retained observed row and every
//! eliminated candidate's collision witness.
use super::*;
fn power(mut x: usize, mut n: usize, p: usize) -> usize {
    let mut result = 1;
    while n > 0 {
        if n % 2 == 1 {
            result = result * x % p;
        }
        x = x * x % p;
        n /= 2;
    }
    result
}
fn term(e: [i8; 2], a: usize, b: usize, p: usize) -> usize {
    let mut result = 1;
    for (x, k) in [a, b].into_iter().zip(e) {
        let base = if k < 0 {
            if x == 0 { 0 } else { power(x, p - 2, p) }
        } else {
            x
        };
        let mut component = 1;
        for _ in 0..k.unsigned_abs() {
            component = component * base % p;
        }
        result = result * component % p;
    }
    result
}
fn quotient(mut numerator: usize, denominator: usize) -> usize {
    if denominator == 0 {
        return 0;
    }
    let mut result = 0;
    while numerator >= denominator {
        numerator -= denominator;
        result += 1;
    }
    result
}
pub(super) fn evaluate(program: Program, a: usize, b: usize, p: usize) -> usize {
    match program {
        Program::Monomial { a: i, b: j } => term([i, j], a, b, p),
        Program::Sum {
            left,
            right,
            coefficient,
        } => {
            let x = term(left, a, b, p);
            let y = term(right, a, b, p);
            if coefficient > 0 {
                (x + coefficient as usize * y) % p
            } else {
                (x + 2 * p - coefficient.unsigned_abs() as usize * y % p) % p
            }
        }
        Program::Integer(op) => {
            use Integer::*;
            let result = match op {
                Min => {
                    if a < b {
                        a
                    } else {
                        b
                    }
                }
                Max => {
                    if a > b {
                        a
                    } else {
                        b
                    }
                }
                Distance => (a as i64 - b as i64).unsigned_abs() as usize,
                QuotientAB => quotient(a, b),
                QuotientBA => quotient(b, a),
                RemainderAB => a - quotient(a, b) * b,
                RemainderBA => b - quotient(b, a) * a,
                And => a & b,
                Or => a | b,
                Xor => a ^ b,
                Equal => (a == b) as usize,
                Less => (a < b) as usize,
                Greater => (a > b) as usize,
                Gcd => {
                    if a == 0 && b == 0 {
                        0
                    } else {
                        (1..=a.max(b))
                            .rev()
                            .find(|d| a.is_multiple_of(*d) && b.is_multiple_of(*d))
                            .unwrap()
                    }
                }
            };
            result % p
        }
    }
}

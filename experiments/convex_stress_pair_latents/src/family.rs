//! All programs are fixed before labels: 49 Laurent monomials of two input
//! coordinates, every ordered-index two-term sum with coefficient +/-1,+/-2,
//! and14 ordinary integer primitives. There is no target AST or task switch.
use crate::*;
mod reference;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Integer {
    Min,
    Max,
    Distance,
    QuotientAB,
    QuotientBA,
    RemainderAB,
    RemainderBA,
    And,
    Or,
    Xor,
    Equal,
    Less,
    Greater,
    Gcd,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Program {
    Monomial {
        a: i8,
        b: i8,
    },
    Sum {
        left: [i8; 2],
        right: [i8; 2],
        coefficient: i8,
    },
    Integer(Integer),
}
pub fn programs() -> Vec<Program> {
    let terms: Vec<_> = (-3..=3)
        .flat_map(|a| (-3..=3).map(move |b| [a, b]))
        .collect();
    let mut out = Vec::with_capacity(PROGRAMS);
    for &[a, b] in &terms {
        out.push(Program::Monomial { a, b });
    }
    for i in 0..terms.len() {
        for j in i + 1..terms.len() {
            for coefficient in [-2, -1, 1, 2] {
                out.push(Program::Sum {
                    left: terms[i],
                    right: terms[j],
                    coefficient,
                });
            }
        }
    }
    use Integer::*;
    for op in [
        Min,
        Max,
        Distance,
        QuotientAB,
        QuotientBA,
        RemainderAB,
        RemainderBA,
        And,
        Or,
        Xor,
        Equal,
        Less,
        Greater,
        Gcd,
    ] {
        out.push(Program::Integer(op));
    }
    assert_eq!(out.len(), PROGRAMS);
    out
}
pub struct Evaluator {
    pub vocabulary: usize,
    powers: Vec<[usize; 7]>,
}
fn inverse(a: usize, p: usize) -> usize {
    if a == 0 {
        return 0;
    }
    let (mut r, mut next_r) = (p as i64, a as i64);
    let (mut t, mut next_t) = (0i64, 1i64);
    while next_r != 0 {
        let q = r / next_r;
        (r, next_r) = (next_r, r - q * next_r);
        (t, next_t) = (next_t, t - q * next_t);
    }
    t.rem_euclid(p as i64) as usize
}
impl Evaluator {
    pub fn new(p: usize) -> Result<Self> {
        if !(2..=1024).contains(&p) || (2..p).any(|d| p.is_multiple_of(d)) {
            return Err("pair family requires a prime vocabulary at most 1024".into());
        }
        let mut powers = Vec::with_capacity(p);
        for x in 0..p {
            let y = inverse(x, p);
            powers.push([
                y * y % p * y % p,
                y * y % p,
                y,
                1,
                x,
                x * x % p,
                x * x % p * x % p,
            ]);
        }
        Ok(Self {
            vocabulary: p,
            powers,
        })
    }
    fn term(&self, [i, j]: [i8; 2], a: usize, b: usize) -> usize {
        self.powers[a][(i + 3) as usize] * self.powers[b][(j + 3) as usize] % self.vocabulary
    }
    pub fn evaluate(&self, program: Program, a: usize, b: usize) -> usize {
        let p = self.vocabulary;
        match program {
            Program::Monomial { a: i, b: j } => self.term([i, j], a, b),
            Program::Sum {
                left,
                right,
                coefficient,
            } => {
                (self.term(left, a, b) as i64 + coefficient as i64 * self.term(right, a, b) as i64)
                    .rem_euclid(p as i64) as usize
            }
            Program::Integer(op) => {
                use Integer::*;
                (match op {
                    Min => a.min(b),
                    Max => a.max(b),
                    Distance => a.abs_diff(b),
                    QuotientAB => a.checked_div(b).unwrap_or(0),
                    QuotientBA => b.checked_div(a).unwrap_or(0),
                    RemainderAB => a.checked_rem(b).unwrap_or(a),
                    RemainderBA => b.checked_rem(a).unwrap_or(b),
                    And => a & b,
                    Or => a | b,
                    Xor => a ^ b,
                    Equal => usize::from(a == b),
                    Less => usize::from(a < b),
                    Greater => usize::from(a > b),
                    Gcd => {
                        let (mut x, mut y) = (a, b);
                        while y != 0 {
                            (x, y) = (y, x % y);
                        }
                        x
                    }
                }) % p
            }
        }
    }
    /// Independent interpreter uses repeated powers and Fermat inversion,
    /// never the evaluator's power table or extended-Euclid inverse.
    pub fn reference(&self, program: Program, a: usize, b: usize) -> usize {
        reference::evaluate(program, a, b, self.vocabulary)
    }
}

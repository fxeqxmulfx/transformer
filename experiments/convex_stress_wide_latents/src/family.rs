//! The full previous bank, followed by all 294784 three-term extensions.
//! Primary arithmetic uses tables and a signed sum; the independent path
//! uses direct powers/Fermat inversion and unsigned repeated additions.
use crate::*;
use pairs::family::{Evaluator as BaseEvaluator, Program as Base};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum Program {
    Base(Base),
    Triple {
        terms: [[i8; 2]; 3],
        coefficients: [i8; 2],
    },
}
pub fn programs() -> Vec<Program> {
    let mut out = Vec::with_capacity(PROGRAMS);
    out.extend(pairs::family::programs().into_iter().map(Program::Base));
    let terms: Vec<_> = (-3..=3)
        .flat_map(|a| (-3..=3).map(move |b| [a, b]))
        .collect();
    for i in 0..terms.len() {
        for j in i + 1..terms.len() {
            for k in j + 1..terms.len() {
                for c in [-2, -1, 1, 2] {
                    for d in [-2, -1, 1, 2] {
                        out.push(Program::Triple {
                            terms: [terms[i], terms[j], terms[k]],
                            coefficients: [c, d],
                        });
                    }
                }
            }
        }
    }
    assert_eq!(out.len(), PROGRAMS);
    out
}
pub struct Evaluator {
    base: BaseEvaluator,
}
impl Evaluator {
    pub fn new(vocabulary: usize) -> Result<Self> {
        Ok(Self {
            base: BaseEvaluator::new(vocabulary)?,
        })
    }
    pub fn vocabulary(&self) -> usize {
        self.base.vocabulary
    }
    pub fn evaluate(&self, program: Program, a: usize, b: usize) -> usize {
        match program {
            Program::Base(p) => self.base.evaluate(p, a, b),
            Program::Triple {
                terms,
                coefficients,
            } => {
                let mut value = 0i64;
                for ([i, j], c) in terms.into_iter().zip([1, coefficients[0], coefficients[1]]) {
                    value += i64::from(c)
                        * self.base.evaluate(Base::Monomial { a: i, b: j }, a, b) as i64;
                }
                value.rem_euclid(self.vocabulary() as i64) as usize
            }
        }
    }
    pub fn reference(&self, program: Program, a: usize, b: usize) -> usize {
        match program {
            Program::Base(p) => self.base.reference(p, a, b),
            Program::Triple {
                terms,
                coefficients,
            } => {
                let p = self.vocabulary();
                let mut value = 0;
                for (position, [i, j]) in terms.into_iter().enumerate() {
                    let x = self.base.reference(Base::Monomial { a: i, b: j }, a, b);
                    let c = if position == 0 {
                        1
                    } else {
                        coefficients[position - 1]
                    };
                    for _ in 0..c.unsigned_abs() {
                        value = if c > 0 {
                            (value + x) % p
                        } else {
                            (value + p - x) % p
                        };
                    }
                }
                value
            }
        }
    }
    /// Bounds include every direct-power and inversion iteration for p<=1024.
    /// Integer primitives retain a separate linear-p bound for divisor scans
    /// and repeated subtraction; three-term programs never use that shortcut.
    pub fn reference_cost(&self, program: Program) -> u128 {
        match program {
            Program::Base(Base::Monomial { .. }) => 512,
            Program::Base(Base::Sum { .. }) => 1024,
            Program::Base(Base::Integer(_)) => 64 * (self.vocabulary() as u128 + 1),
            Program::Triple { .. } => 1536,
        }
    }
}

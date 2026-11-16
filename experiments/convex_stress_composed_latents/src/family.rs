//! A complete, observation-independent finite composition grammar. Equal
//! functions retain separate candidate blocks; no fitted beam or top-k exists.
use crate::*;
use pairs::family::{Evaluator as PairEvaluator, Program as Pair};
mod reference;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Input {
    A,
    B,
    Sum,
    Product,
    Distance,
}
pub const INPUTS: [Input; 5] = [
    Input::A,
    Input::B,
    Input::Sum,
    Input::Product,
    Input::Distance,
];
impl Input {
    fn value(self, a: usize, b: usize) -> usize {
        match self {
            Self::A => a,
            Self::B => b,
            Self::Sum => a + b,
            Self::Product => a * b,
            Self::Distance => a.abs_diff(b),
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Leaf {
    pub source: Pair,
    pub sign: i8,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum Program {
    Old(wide::family::Program),
    FourTerms {
        terms: [[i8; 2]; 4],
        coefficients: [i8; 3],
    },
    Branch {
        control: Input,
        modulus: u8,
        threshold: u8,
        low: Leaf,
        high: Leaf,
    },
    Recombine {
        source: Pair,
        control: Input,
        modulus: u8,
    },
    Digit {
        source: Input,
        high: bool,
    },
}
pub fn leaves() -> Vec<Leaf> {
    let mut source: Vec<_> = (-3..=3)
        .flat_map(|a| (-3..=3).map(move |b| Pair::Monomial { a, b }))
        .collect();
    let basis = [[-1, 0], [0, -1], [0, 0], [0, 1], [1, 0]];
    for i in 0..basis.len() {
        for j in i + 1..basis.len() {
            for coefficient in [-2, -1, 1, 2] {
                source.push(Pair::Sum {
                    left: basis[i],
                    right: basis[j],
                    coefficient,
                });
            }
        }
    }
    source
        .into_iter()
        .flat_map(|source| [-1, 1].map(|sign| Leaf { source, sign }))
        .collect()
}
pub fn programs() -> Vec<Program> {
    let mut out = Vec::with_capacity(PROGRAMS);
    out.extend(wide::family::programs().into_iter().map(Program::Old));
    let terms: Vec<_> = (0..=3).flat_map(|a| (0..=3).map(move |b| [a, b])).collect();
    for i in 0..terms.len() {
        for j in i + 1..terms.len() {
            for k in j + 1..terms.len() {
                for l in k + 1..terms.len() {
                    for c in [-2, -1, 1, 2] {
                        for d in [-2, -1, 1, 2] {
                            for e in [-2, -1, 1, 2] {
                                out.push(Program::FourTerms {
                                    terms: [terms[i], terms[j], terms[k], terms[l]],
                                    coefficients: [c, d, e],
                                });
                            }
                        }
                    }
                }
            }
        }
    }
    let leaves = leaves();
    for control in INPUTS {
        for modulus in [2, 3] {
            for threshold in 1..modulus {
                for &low in &leaves {
                    for &high in &leaves {
                        out.push(Program::Branch {
                            control,
                            modulus,
                            threshold,
                            low,
                            high,
                        });
                    }
                }
            }
        }
    }
    for source in pairs::family::programs() {
        for control in INPUTS {
            for modulus in [2, 3, 4] {
                out.push(Program::Recombine {
                    source,
                    control,
                    modulus,
                });
            }
        }
    }
    for source in INPUTS {
        for high in [false, true] {
            out.push(Program::Digit { source, high });
        }
    }
    assert_eq!(out.len(), PROGRAMS);
    out
}
pub struct Evaluator {
    old: wide::family::Evaluator,
    pair: PairEvaluator,
}
impl Evaluator {
    pub fn new(vocabulary: usize) -> Result<Self> {
        Ok(Self {
            old: wide::family::Evaluator::new(vocabulary)?,
            pair: PairEvaluator::new(vocabulary)?,
        })
    }
    pub fn vocabulary(&self) -> usize {
        self.old.vocabulary()
    }
    pub fn evaluate(&self, program: Program, a: usize, b: usize) -> usize {
        let p = self.vocabulary();
        match program {
            Program::Old(tree) => self.old.evaluate(tree, a, b),
            Program::FourTerms {
                terms,
                coefficients,
            } => {
                let mut value = 0i64;
                for ([i, j], c) in
                    terms
                        .into_iter()
                        .zip([1, coefficients[0], coefficients[1], coefficients[2]])
                {
                    value += i64::from(c)
                        * self.pair.evaluate(Pair::Monomial { a: i, b: j }, a, b) as i64;
                }
                value.rem_euclid(p as i64) as usize
            }
            Program::Branch {
                control,
                modulus,
                threshold,
                low,
                high,
            } => {
                let leaf = if control.value(a, b) % usize::from(modulus) < usize::from(threshold) {
                    low
                } else {
                    high
                };
                let value = self.pair.evaluate(leaf.source, a, b);
                if leaf.sign < 0 {
                    (p - value) % p
                } else {
                    value
                }
            }
            Program::Recombine {
                source,
                control,
                modulus,
            } => {
                let q = usize::from(modulus);
                (q * (self.pair.evaluate(source, a, b) / q) + control.value(a, b) % q) % p
            }
            Program::Digit { source, high } => {
                let value = source.value(a, b);
                (if high { value / p } else { value }) % p
            }
        }
    }
    pub fn reference(&self, program: Program, a: usize, b: usize) -> usize {
        reference::evaluate(self, program, a, b)
    }
    /// Charges the entire independent path, including bounded subtraction.
    /// A branch charges the more expensive leaf, even if the other is selected.
    pub fn reference_cost(&self, program: Program) -> u128 {
        let pair = |tree| self.old.reference_cost(wide::family::Program::Base(tree));
        match program {
            Program::Old(tree) => self.old.reference_cost(tree),
            Program::FourTerms { .. } => 2112,
            Program::Branch { low, high, .. } => 64 + pair(low.source).max(pair(high.source)),
            Program::Recombine { source, .. } => {
                pair(source) + 8 * (self.vocabulary() as u128 + 1) + 64
            }
            Program::Digit { .. } => 32 * (self.vocabulary() as u128 + 1),
        }
    }
}

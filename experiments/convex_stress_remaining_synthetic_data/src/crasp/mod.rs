//! Frozen synthetic/crasp.py inclusive past-counting formulas and seeded
//! nonconstant-program selection. This is a data oracle, never model input.
use crate::*;
use serde::{Deserialize, Serialize};
mod formula;
mod independent;
mod program;
pub use program::{Selected, select};
pub const SYMBOLS: [i64; 3] = [9, 10, 23];

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Term {
    Const(i64),
    Count(Box<Formula>),
    Neg(Box<Term>),
    Add(Box<Term>, Box<Term>),
}
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Formula {
    Sym(i64),
    Lt(Term, Term),
    Not(Box<Formula>),
    And(Box<Formula>, Box<Formula>),
}
impl Formula {
    pub fn evaluate(&self, word: &[i64]) -> Result<Vec<bool>> {
        if word.len() > 4096
            || word.iter().any(|x| !SYMBOLS.contains(x))
            || self.nodes() > 4096
            || self.depth() > 5
        {
            return Err("formula evaluation dimensions or alphabet refused".into());
        }
        self.eval(word)
    }
    pub fn independent(&self, word: &[i64]) -> Result<Vec<bool>> {
        if word.len() > 4096
            || word.iter().any(|x| !SYMBOLS.contains(x))
            || self.nodes() > 4096
            || self.depth() > 5
        {
            return Err("independent formula dimensions or alphabet refused".into());
        }
        independent::evaluate(self, word)
    }
}

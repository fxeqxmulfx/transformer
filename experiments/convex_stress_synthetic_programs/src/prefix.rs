//! Independently implemented bounded prefix operator grammar. It deliberately
//! does not import the dataset's formula type, generator, seed or target AST.
use crate::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Term {
    Constant(i32),
    Count(Box<Formula>),
    Negate(Box<Term>),
    Add(Box<Term>, Box<Term>),
}
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Formula {
    Symbol(usize),
    Less(Box<Term>, Box<Term>),
    Not(Box<Formula>),
    And(Box<Formula>, Box<Formula>),
}

impl Term {
    pub fn nodes(&self) -> usize {
        1 + match self {
            Self::Constant(_) => 0,
            Self::Count(f) => f.nodes(),
            Self::Negate(t) => t.nodes(),
            Self::Add(a, b) => a.nodes() + b.nodes(),
        }
    }
    fn eval(&self, word: &[usize]) -> Result<Vec<i32>> {
        Ok(match self {
            Self::Constant(n) => vec![*n; word.len()],
            Self::Count(f) => {
                let mut n = 0i32;
                f.eval(word)?
                    .into_iter()
                    .map(|x| {
                        n += i32::from(x);
                        n
                    })
                    .collect()
            }
            Self::Negate(t) => t
                .eval(word)?
                .into_iter()
                .map(|x| {
                    x.checked_neg()
                        .ok_or_else(|| "prefix negation overflow".into())
                })
                .collect::<Result<_>>()?,
            Self::Add(a, b) => a
                .eval(word)?
                .into_iter()
                .zip(b.eval(word)?)
                .map(|(a, b)| {
                    a.checked_add(b)
                        .ok_or_else(|| "prefix addition overflow".into())
                })
                .collect::<Result<_>>()?,
        })
    }
}
impl Formula {
    pub fn nodes(&self) -> usize {
        1 + match self {
            Self::Symbol(_) => 0,
            Self::Less(a, b) => a.nodes() + b.nodes(),
            Self::Not(f) => f.nodes(),
            Self::And(a, b) => a.nodes() + b.nodes(),
        }
    }
    fn eval(&self, word: &[usize]) -> Result<Vec<bool>> {
        Ok(match self {
            Self::Symbol(s) => word.iter().map(|x| x == s).collect(),
            Self::Less(a, b) => a
                .eval(word)?
                .iter()
                .zip(b.eval(word)?)
                .map(|(a, b)| *a < b)
                .collect(),
            Self::Not(f) => f.eval(word)?.into_iter().map(|x| !x).collect(),
            Self::And(a, b) => a
                .eval(word)?
                .iter()
                .zip(b.eval(word)?)
                .map(|(a, b)| *a && b)
                .collect(),
        })
    }
    pub fn evaluate(&self, word: &[usize]) -> Result<Vec<bool>> {
        if word.is_empty()
            || word.len() > 4096
            || self.nodes() > 127
            || word.iter().any(|x| ![9, 10, 23].contains(x))
        {
            return Err("prefix program dimensions or supplied alphabet refused".into());
        }
        self.eval(word)
    }
}

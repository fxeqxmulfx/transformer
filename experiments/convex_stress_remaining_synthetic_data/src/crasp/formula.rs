//! Recursive vector evaluation, including inclusive prefix-sum counts.
use super::*;
impl Term {
    pub fn depth(&self) -> usize {
        match self {
            Self::Const(_) => 0,
            Self::Count(f) => 1 + f.depth(),
            Self::Neg(t) => t.depth(),
            Self::Add(a, b) => a.depth().max(b.depth()),
        }
    }
    fn nodes(&self) -> usize {
        1 + match self {
            Self::Const(_) => 0,
            Self::Count(f) => f.nodes(),
            Self::Neg(t) => t.nodes(),
            Self::Add(a, b) => a.nodes() + b.nodes(),
        }
    }
    pub fn describe(&self) -> String {
        match self {
            Self::Const(n) => n.to_string(),
            Self::Count(f) => format!("count({})", f.describe()),
            Self::Neg(t) => format!("neg({})", t.describe()),
            Self::Add(a, b) => format!("add({}, {})", a.describe(), b.describe()),
        }
    }
    fn eval(&self, word: &[i64]) -> Result<Vec<i64>> {
        Ok(match self {
            Self::Const(n) => vec![*n; word.len()],
            Self::Count(f) => {
                let mut sum = 0;
                f.eval(word)?
                    .into_iter()
                    .map(|b| {
                        sum += i64::from(b);
                        sum
                    })
                    .collect()
            }
            Self::Neg(t) => t
                .eval(word)?
                .iter()
                .map(|x| x.checked_neg().ok_or("term negation overflow"))
                .collect::<std::result::Result<_, _>>()?,
            Self::Add(a, b) => a
                .eval(word)?
                .iter()
                .zip(b.eval(word)?)
                .map(|(a, b)| a.checked_add(b).ok_or("term sum overflow"))
                .collect::<std::result::Result<_, _>>()?,
        })
    }
}
impl Formula {
    pub fn depth(&self) -> usize {
        match self {
            Self::Sym(_) => 0,
            Self::Lt(a, b) => a.depth().max(b.depth()),
            Self::Not(f) => f.depth(),
            Self::And(a, b) => a.depth().max(b.depth()),
        }
    }
    pub(super) fn nodes(&self) -> usize {
        1 + match self {
            Self::Sym(_) => 0,
            Self::Lt(a, b) => a.nodes() + b.nodes(),
            Self::Not(f) => f.nodes(),
            Self::And(a, b) => a.nodes() + b.nodes(),
        }
    }
    pub fn describe(&self) -> String {
        match self {
            Self::Sym(s) => match s {
                9 => "a",
                10 => "b",
                23 => "c",
                _ => "invalid",
            }
            .into(),
            Self::Lt(a, b) => format!("lt({}, {})", a.describe(), b.describe()),
            Self::Not(f) => format!("not({})", f.describe()),
            Self::And(a, b) => format!("and({}, {})", a.describe(), b.describe()),
        }
    }
    pub(super) fn eval(&self, word: &[i64]) -> Result<Vec<bool>> {
        Ok(match self {
            Self::Sym(s) => {
                if !SYMBOLS.contains(s) {
                    return Err("invalid formula symbol".into());
                }
                word.iter().map(|x| x == s).collect()
            }
            Self::Lt(a, b) => a
                .eval(word)?
                .iter()
                .zip(b.eval(word)?)
                .map(|(a, b)| *a < b)
                .collect(),
            Self::Not(f) => f.eval(word)?.iter().map(|x| !x).collect(),
            Self::And(a, b) => a
                .eval(word)?
                .iter()
                .zip(b.eval(word)?)
                .map(|(a, b)| *a && b)
                .collect(),
        })
    }
}

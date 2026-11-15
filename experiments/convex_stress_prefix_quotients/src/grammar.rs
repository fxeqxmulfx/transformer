//! Supplied fixed grammar, matching343 plus explicit Boolean constants.
//! No program family is selected from a dataset task name.
use crate::{Formula, Term};

pub fn literal(i: usize) -> Formula {
    let f = Formula::Symbol([9, 10, 23][i % 3]);
    if i < 3 { f } else { Formula::Not(Box::new(f)) }
}
pub fn constants(value: bool) -> Formula {
    Formula::Less(
        Box::new(Term::Constant(0)),
        Box::new(Term::Constant(i32::from(value))),
    )
}
pub fn base() -> Vec<Formula> {
    (0..6)
        .map(literal)
        .chain([constants(false), constants(true)])
        .collect()
}
pub fn make(
    a: &Formula,
    b: &Formula,
    extra: Option<usize>,
    offset: i32,
    negated: bool,
    gate: Option<usize>,
) -> Formula {
    let left = Term::Count(Box::new(a.clone()));
    let mut right = Term::Count(Box::new(b.clone()));
    if let Some(s) = extra {
        right = Term::Add(Box::new(right), Box::new(Term::Count(Box::new(literal(s)))));
    }
    if offset != 0 {
        right = Term::Add(Box::new(right), Box::new(Term::Constant(offset)));
    }
    let mut f = Formula::Less(Box::new(left), Box::new(right));
    if negated {
        f = Formula::Not(Box::new(f));
    }
    if let Some(g) = gate {
        f = Formula::And(Box::new(f), Box::new(literal(g)));
    }
    f
}
pub fn size(
    a: usize,
    b: usize,
    extra: Option<usize>,
    offset: i32,
    negated: bool,
    gate: Option<usize>,
) -> usize {
    3 + a
        + b
        + usize::from(extra.is_some()) * 3
        + usize::from(offset != 0) * 2
        + usize::from(negated)
        + gate.map_or(0, |g| if g < 3 { 2 } else { 3 })
}

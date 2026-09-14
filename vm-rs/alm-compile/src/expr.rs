//! Linear expressions over graph dimensions.
//!
//! A port of `Expression` in `transformer_vm/graph/core.py`.  The one thing
//! that is not incidental here is the *order* of the terms: Python holds them
//! in a `dict`, insertion-ordered, and the weight builder sums coefficients
//! into a residual slot in exactly that order (two dimensions can share a slot
//! after reuse, and float addition is not commutative once it rounds).  So the
//! terms live in a `Vec` and every operation reproduces CPython's dict
//! behaviour: an update keeps a key in place, a deletion removes it, and a
//! re-insertion appends at the end.

/// Index of a dimension in the graph's arena — also its `Dimension.id`.
pub type DimId = u32;

/// A canonical, hashable form of an expression, used as a memoization key.
/// `_expr_key` in the original sorts by `id(dim)`; sorting by the dimension's
/// own id is the same equivalence, and is reproducible across runs.
pub type ExprKey = Vec<(DimId, u64)>;

/// A linear combination of dimensions.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Expr {
    terms: Vec<(DimId, f64)>,
}

impl Expr {
    pub fn new() -> Self {
        Expr { terms: Vec::new() }
    }

    /// A single dimension with coefficient 1.
    pub fn dim(d: DimId) -> Self {
        Expr { terms: vec![(d, 1.0)] }
    }

    /// `coeff * dim`, empty when the coefficient is zero (the Python
    /// constructor prunes zeros, so `0 * d` and the empty expression are the
    /// same object to every consumer).
    pub fn scaled(d: DimId, coeff: f64) -> Self {
        if coeff == 0.0 {
            Expr::new()
        } else {
            Expr { terms: vec![(d, coeff)] }
        }
    }

    pub fn from_terms(terms: Vec<(DimId, f64)>) -> Self {
        Expr { terms: terms.into_iter().filter(|&(_, c)| c != 0.0).collect() }
    }

    pub fn terms(&self) -> &[(DimId, f64)] {
        &self.terms
    }

    pub fn len(&self) -> usize {
        self.terms.len()
    }

    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    pub fn get(&self, d: DimId) -> f64 {
        self.terms.iter().find(|&&(k, _)| k == d).map_or(0.0, |&(_, c)| c)
    }

    pub fn contains(&self, d: DimId) -> bool {
        self.terms.iter().any(|&(k, _)| k == d)
    }

    /// `expr[dim] = value`, with Python's semantics: zero deletes, and a
    /// re-inserted key lands at the end.
    pub fn set(&mut self, d: DimId, value: f64) {
        match self.terms.iter().position(|&(k, _)| k == d) {
            Some(i) => {
                if value == 0.0 {
                    self.terms.remove(i);
                } else {
                    self.terms[i].1 = value;
                }
            }
            None => {
                if value != 0.0 {
                    self.terms.push((d, value));
                }
            }
        }
    }

    fn accumulate(&mut self, d: DimId, delta: f64) {
        match self.terms.iter().position(|&(k, _)| k == d) {
            Some(i) => {
                let v = self.terms[i].1 + delta;
                if v == 0.0 {
                    self.terms.remove(i);
                } else {
                    self.terms[i].1 = v;
                }
            }
            None => {
                if delta != 0.0 {
                    self.terms.push((d, delta));
                }
            }
        }
    }

    pub fn add(&self, other: &Expr) -> Expr {
        let mut r = self.clone();
        for &(d, c) in &other.terms {
            r.accumulate(d, c);
        }
        r
    }

    pub fn sub(&self, other: &Expr) -> Expr {
        let mut r = self.clone();
        for &(d, c) in &other.terms {
            r.accumulate(d, -c);
        }
        r
    }

    pub fn neg(&self) -> Expr {
        Expr { terms: self.terms.iter().map(|&(d, c)| (d, -c)).collect() }
    }

    pub fn mul(&self, k: f64) -> Expr {
        if k == 0.0 {
            return Expr::new();
        }
        Expr::from_terms(self.terms.iter().map(|&(d, c)| (d, c * k)).collect())
    }

    pub fn key(&self) -> ExprKey {
        let mut k: ExprKey = self.terms.iter().map(|&(d, c)| (d, c.to_bits())).collect();
        k.sort_unstable();
        k
    }

    /// `sum(c * values[d])`, in term order.
    pub fn evaluate(&self, values: &dyn Fn(DimId) -> f64) -> f64 {
        self.terms.iter().map(|&(d, c)| c * values(d)).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_coefficients_never_enter_an_expression() {
        assert!(Expr::scaled(7, 0.0).is_empty());
        assert!(Expr::from_terms(vec![(1, 0.0), (2, -0.0)]).is_empty());
        assert!(Expr::dim(3).mul(0.0).is_empty());
    }

    #[test]
    fn cancelling_a_term_removes_it_and_re_adding_appends() {
        let a = Expr::from_terms(vec![(1, 1.0), (2, 1.0)]);
        let b = a.sub(&Expr::dim(1));
        assert_eq!(b.terms(), &[(2, 1.0)]);
        let c = b.add(&Expr::dim(1));
        assert_eq!(c.terms(), &[(2, 1.0), (1, 1.0)], "re-insertion goes to the end");
    }

    #[test]
    fn updating_a_live_term_keeps_its_place() {
        let a = Expr::from_terms(vec![(1, 1.0), (2, 1.0)]);
        let b = a.add(&Expr::scaled(1, 4.0));
        assert_eq!(b.terms(), &[(1, 5.0), (2, 1.0)]);
    }

    #[test]
    fn the_key_is_order_free() {
        let a = Expr::from_terms(vec![(1, 2.0), (5, -1.0)]);
        let b = Expr::from_terms(vec![(5, -1.0), (1, 2.0)]);
        assert_ne!(a.terms(), b.terms());
        assert_eq!(a.key(), b.key());
    }
}

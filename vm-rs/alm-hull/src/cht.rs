//! The dynamic convex hull trick: an upper envelope of lines under insertion.
//!
//! Ported from `_HullCHT` in `attention/hull2d_cht.h`.  The C++ keeps the
//! envelope in one `std::multiset` ordered by slope and searches it by
//! breakpoint through a heterogeneous comparator, which works because the
//! breakpoints increase with the slope.  Rust's `BTreeMap` has no cursor API on
//! stable, so the same invariant is carried by a second index, ordered by
//! breakpoint; every mutation goes through `set_break` and `erase`, which keep
//! the two in step.  Both orders are total and exact, so the structure holds
//! the same envelope the C++ does, line for line.

use crate::breakpoint::Break;
use crate::meta::HullMeta;
use core::cmp::Ordering;
use core::ops::Bound::{Excluded, Unbounded};
use std::collections::{BTreeMap, BTreeSet};

/// A slope, ordered totally so that it can key a map.
///
/// Construction folds `-0.0` to `0.0`: the lower envelope stores negated keys,
/// so a zero slope arrives with either sign, and the two must name one line.
#[derive(Clone, Copy, Debug)]
pub struct Slope(f64);

impl Slope {
    pub fn new(m: f64) -> Slope {
        Slope(if m == 0.0 { 0.0 } else { m })
    }
    pub fn get(self) -> f64 {
        self.0
    }
    fn least() -> Slope {
        Slope(f64::NEG_INFINITY)
    }
}

impl PartialEq for Slope {
    fn eq(&self, o: &Self) -> bool {
        self.0.total_cmp(&o.0) == Ordering::Equal
    }
}
impl Eq for Slope {}
impl PartialOrd for Slope {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Slope {
    fn cmp(&self, o: &Self) -> Ordering {
        self.0.total_cmp(&o.0)
    }
}

/// One line of the envelope: `y = m x + b`, optimal up to `p`.
#[derive(Clone, Copy, Debug)]
pub struct Line {
    pub b: f64,
    pub p: Break,
    pub meta: HullMeta,
}

/// The upper envelope of a set of lines.
#[derive(Default)]
pub struct Cht {
    lines: BTreeMap<Slope, Line>,
    breaks: BTreeSet<(Break, Slope)>,
}

impl Cht {
    pub fn new() -> Cht {
        Cht::default()
    }

    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }

    pub fn len(&self) -> usize {
        self.lines.len()
    }

    pub fn clear(&mut self) {
        self.lines.clear();
        self.breaks.clear();
    }

    pub fn get(&self, s: Slope) -> Option<&Line> {
        self.lines.get(&s)
    }

    pub fn succ(&self, s: Slope) -> Option<Slope> {
        self.lines.range((Excluded(s), Unbounded)).next().map(|(k, _)| *k)
    }

    pub fn pred(&self, s: Slope) -> Option<Slope> {
        self.lines.range((Unbounded, Excluded(s))).next_back().map(|(k, _)| *k)
    }

    fn set_break(&mut self, s: Slope, p: Break) {
        let old = self.lines[&s].p;
        if old != Break::Unset {
            self.breaks.remove(&(old, s));
        }
        self.lines.get_mut(&s).unwrap().p = p;
        if p != Break::Unset {
            self.breaks.insert((p, s));
        }
    }

    fn erase(&mut self, s: Slope) {
        if let Some(l) = self.lines.remove(&s) {
            if l.p != Break::Unset {
                self.breaks.remove(&(l.p, s));
            }
        }
    }

    /// Set `x`'s breakpoint against `y`, and report whether `y` is now useless.
    ///
    /// `y = None` stands for the end of the envelope, where `x` runs forever.
    fn isect(&mut self, x: Slope, y: Option<Slope>) -> bool {
        let Some(ys) = y else {
            self.set_break(x, Break::PosInf);
            return false;
        };
        let (my, by, yp) = {
            let l = &self.lines[&ys];
            (ys.get(), l.b, l.p)
        };
        let (mx, bx) = (x.get(), self.lines[&x].b);
        let p = Break::between(mx, bx, my, by);
        self.set_break(x, p);
        p >= yp
    }

    /// Insert `y = m x + b` carrying `meta`, keeping only the upper envelope.
    pub fn add_line(&mut self, m: f64, b: f64, meta: HullMeta) {
        let s = Slope::new(m);
        let mut new_meta = meta;

        if let Some(l) = self.lines.get(&s) {
            if l.b == b {
                let mut merged = l.meta;
                merged.merge(&meta);
                new_meta = merged;
            } else if l.b >= b {
                return;
            }
            self.erase(s);
        }
        self.lines.insert(s, Line { b, p: Break::Unset, meta: new_meta });

        // Drop successors the new line has made redundant.
        let mut z = self.succ(s);
        while self.isect(s, z) {
            let zs = z.unwrap();
            z = self.succ(zs);
            self.erase(zs);
        }

        // The predecessor may make the new line itself redundant.
        let mut x = s;
        if let Some(px) = self.pred(s) {
            x = px;
            if self.isect(x, Some(s)) {
                let after = self.succ(s);
                self.erase(s);
                self.isect(x, after);
            }
        }

        // Walk left, dropping lines whose breakpoints are out of order.
        loop {
            let y = x;
            let Some(px) = self.pred(y) else { break };
            x = px;
            if self.lines[&x].p >= self.lines[&y].p {
                let after = self.succ(y);
                self.erase(y);
                self.isect(x, after);
            } else {
                break;
            }
        }
    }

    /// The line maximal at `x`: the first whose breakpoint reaches `x`.
    pub fn argmax(&self, x: Break) -> Option<Slope> {
        self.breaks
            .range((x, Slope::least())..)
            .next()
            .map(|(_, s)| *s)
            .or_else(|| self.lines.keys().next_back().copied())
    }

    /// The envelope in increasing slope order, for tests and for debugging.
    pub fn iter(&self) -> impl Iterator<Item = (Slope, &Line)> {
        self.lines.iter().map(|(s, l)| (*s, l))
    }
}

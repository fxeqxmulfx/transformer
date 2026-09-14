//! The same upper envelope as `cht`, kept in the tree instead of a vector.
//!
//! The two differ in one place and it is the erase loops.  The vector reads
//! the array to decide what falls and then rewrites the tail once; the cost
//! is that one rewrite, linear in the tail.  Here nothing moves at all: the
//! lines that fall are unlinked at the cursor that found them, and the walk
//! that finds them is the walk that erases them.  That is the C++ cost model
//! -- `hull2d_cht.h` exists because the Percepta vector version ran into the
//! same memmove -- with one descent per call rather than the two the C++
//! spends, because the insertion point is already in hand when the new line
//! is linked in.

use crate::breakpoint::Break;
use crate::meta::HullMeta;
use crate::tree::{Line, Slope, Tree, NIL};

/// The upper envelope of a set of lines, in increasing slope order.
///
/// Two invariants hold between calls: the slopes strictly increase along the
/// envelope, and so do the breakpoints.  The second is why one tree ordered
/// by slope answers a query stated as a breakpoint.
#[derive(Default)]
pub struct Envelope {
    t: Tree,
}

impl Envelope {
    pub fn new() -> Envelope {
        Envelope::default()
    }

    pub fn is_empty(&self) -> bool {
        self.t.is_empty()
    }

    pub fn len(&self) -> usize {
        self.t.len()
    }

    pub fn clear(&mut self) {
        self.t.clear();
    }

    /// The line at the cursor `i`, which must name a live node.
    pub fn get(&self, i: u32) -> Line {
        self.t.get(i)
    }

    /// The next line along the envelope, or `NIL` past the end.
    pub fn next(&self, i: u32) -> u32 {
        self.t.next(i)
    }

    /// The previous line along the envelope, or `NIL` before the start.
    pub fn prev(&self, i: u32) -> u32 {
        self.t.prev(i)
    }

    /// The breakpoint of the line at `i` against `r`, or `PosInf` past the end.
    fn break_against(&self, i: u32, r: Option<Line>) -> Break {
        let x = self.t.get(i);
        match r {
            Some(y) => Break::between(x.m.get(), x.b, y.m.get(), y.b),
            None => Break::PosInf,
        }
    }

    /// The line at the cursor, or `None` past the end.
    fn line_at(&self, i: u32) -> Option<Line> {
        (i != NIL).then(|| self.t.get(i))
    }

    /// Insert `y = m x + b` carrying `meta`, keeping only the upper envelope.
    ///
    /// One descent, at the top, for the whole call.  Everything after it is a
    /// walk from that cursor: the successors the new line hides are erased as
    /// the walk passes them, and so are the predecessors it hides, and the
    /// new line is linked at the place the walk stopped.
    pub fn add_line(&mut self, m: f64, b: f64, meta: HullMeta) {
        let s = Slope::new(m);
        let mut new_meta = meta;

        // Where the new line goes, and where the tail it does not touch begins.
        let at = self.t.lower_bound_slope(s);
        let mut hi = at;
        if at != NIL && self.t.slope_of(at) == s {
            let l = self.t.get(at);
            if l.b == b {
                let mut merged = l.meta;
                merged.merge(&meta);
                new_meta = merged;
            } else if l.b >= b {
                return;
            }
            hi = self.t.next(at);
        }
        let mut new = Line { m: s, b, p: Break::Unset, meta: new_meta };

        // Drop successors the new line has made redundant.
        loop {
            let Some(succ) = self.line_at(hi) else {
                new.p = Break::PosInf;
                break;
            };
            new.p = Break::between(new.m.get(), new.b, succ.m.get(), succ.b);
            if new.p >= succ.p {
                hi = self.t.next(hi);
            } else {
                break;
            }
        }

        // The predecessor may make the new line itself redundant, and then
        // walk left dropping lines whose breakpoints are out of order.
        let mut keep = true;
        let mut lo = at;
        let pred = self.t.prev(at);
        if pred != NIL {
            let mut cur = pred;
            let mut p = self.break_against(cur, Some(new));
            if p >= new.p {
                keep = false;
                p = self.break_against(cur, self.line_at(hi));
            }
            let right = if keep { Some(new) } else { self.line_at(hi) };
            loop {
                let back = self.t.prev(cur);
                if back == NIL || self.t.break_of(back) < p {
                    break;
                }
                cur = back;
                p = self.break_against(cur, right);
            }
            self.t.set_break(cur, p);
            lo = self.t.next(cur);
        }

        // Unlink what fell, then link the new line where the walk stopped.
        // `hi` is a cursor into the tree and survives all of this: erasing a
        // node relinks around it rather than filling it in from a neighbour.
        let mut x = lo;
        while x != hi {
            let next = self.t.next(x);
            self.t.erase(x);
            x = next;
        }
        if keep {
            self.t.insert_before(hi, new);
        }
    }

    /// The line maximal at `x`: the first whose breakpoint reaches `x`.
    ///
    /// The breakpoints increase along the envelope, so the tree ordered by
    /// slope answers this without a second index -- which is what the C++
    /// heterogeneous comparator buys and what this reproduces.
    pub fn argmax(&self, x: Break) -> Option<u32> {
        if self.t.is_empty() {
            return None;
        }
        let at = self.t.lower_bound_break(x);
        Some(if at == NIL { self.t.last() } else { at })
    }

    /// The envelope in increasing slope order, for tests and for debugging.
    pub fn iter(&self) -> impl Iterator<Item = Line> + '_ {
        let mut x = self.t.first();
        core::iter::from_fn(move || {
            (x != NIL).then(|| {
                let l = self.t.get(x);
                x = self.t.next(x);
                l
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cht::Cht;

    /// The envelope's two invariants, which both searches rest on.
    fn well_formed(e: &Envelope) {
        let lines: Vec<Line> = e.iter().collect();
        for w in lines.windows(2) {
            assert!(w[0].m < w[1].m, "slopes increase: {:?} {:?}", w[0].m, w[1].m);
            assert!(w[0].p < w[1].p, "breakpoints increase: {:?} {:?}", w[0].p, w[1].p);
        }
        if let Some(l) = lines.last() {
            assert_eq!(l.p, Break::PosInf, "the last breakpoint is not at infinity");
        }
    }

    /// The tree and the vector must agree line for line, not merely in what
    /// they answer: the same set survives, with the same breakpoints.
    fn same(e: &Envelope, c: &Cht) {
        let got: Vec<Line> = e.iter().collect();
        let want: Vec<&Line> = c.iter().collect();
        assert_eq!(got.len(), want.len(), "envelopes differ in size");
        for (g, w) in got.iter().zip(want) {
            assert_eq!(g.m, w.m, "slopes differ");
            assert_eq!(g.b, w.b, "intercepts differ");
            assert_eq!(g.p, w.p, "breakpoints differ");
        }
    }

    fn rng() -> impl FnMut() -> u64 {
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        }
    }

    /// Random lines, the order that exercises every branch of `add_line`.
    #[test]
    fn agrees_with_the_vector_on_random_lines() {
        let mut next = rng();
        for trial in 0..60 {
            let (mut e, mut c) = (Envelope::new(), Cht::new());
            for _ in 0..200 {
                let m = (next() % 41) as f64 - 20.0;
                let b = (next() % 41) as f64 - 20.0;
                e.add_line(m, b, HullMeta::default());
                c.add_line(m, b, HullMeta::default());
                same(&e, &c);
                well_formed(&e);
            }
            assert!(!e.is_empty(), "trial {trial} emptied the envelope");
        }
    }

    /// The parabolic lift, which is the only shape the engine ever inserts.
    #[test]
    fn agrees_with_the_vector_on_lifted_keys() {
        let mut next = rng();
        for &span in &[16i64, 1024, 65536] {
            let (mut e, mut c) = (Envelope::new(), Cht::new());
            for _ in 0..500 {
                let k = (next() % (span as u64)) as f64;
                let (m, b) = (2.0 * k, -k * k);
                e.add_line(m, b, HullMeta::default());
                c.add_line(m, b, HullMeta::default());
            }
            same(&e, &c);
            well_formed(&e);
        }
    }

    /// The arrival orders `alm-stress` drives, where the vector is quadratic.
    #[test]
    fn agrees_with_the_vector_on_every_arrival_order() {
        for &step in &[1i64, -1, 7, -7, 2003] {
            let (mut e, mut c) = (Envelope::new(), Cht::new());
            for i in 0..600i64 {
                let k = (i * step).rem_euclid(601) as f64;
                let (m, b) = (2.0 * k, -k * k);
                e.add_line(m, b, HullMeta::default());
                c.add_line(m, b, HullMeta::default());
            }
            same(&e, &c);
            well_formed(&e);
        }
    }

    /// Queries, which is the other search and the one the tree must not lose.
    #[test]
    fn argmax_agrees_with_the_vector() {
        let mut next = rng();
        let (mut e, mut c) = (Envelope::new(), Cht::new());
        for _ in 0..400 {
            let k = (next() % 4096) as f64;
            let (m, b) = (2.0 * k, -k * k);
            e.add_line(m, b, HullMeta::default());
            c.add_line(m, b, HullMeta::default());
        }
        for q in 0..4096 {
            let x = Break::between(0.0, 0.0, 1.0, -f64::from(q));
            let a = e.argmax(x).map(|i| e.get(i));
            let b = c.argmax(x).map(|i| *c.get(i));
            match (a, b) {
                (Some(a), Some(b)) => {
                    assert_eq!(a.m, b.m, "argmax differs at {q}");
                    assert_eq!(a.b, b.b, "argmax differs at {q}");
                }
                (None, None) => {}
                _ => panic!("one container answered and the other did not"),
            }
        }
    }

    /// Equal slopes, which is the branch that merges or drops instead.
    #[test]
    fn agrees_with_the_vector_on_repeated_slopes() {
        let mut next = rng();
        let (mut e, mut c) = (Envelope::new(), Cht::new());
        for _ in 0..800 {
            let m = (next() % 7) as f64;
            let b = (next() % 11) as f64 - 5.0;
            e.add_line(m, b, HullMeta::default());
            c.add_line(m, b, HullMeta::default());
            same(&e, &c);
        }
        well_formed(&e);
    }
}

//! The dynamic convex hull trick: an upper envelope of lines under insertion.
//!
//! Ported from `_HullCHT` in `attention/hull2d_cht.h`.  The C++ keeps the
//! envelope in one `std::multiset` ordered by slope and searches it by
//! breakpoint through a heterogeneous comparator, which works because the
//! breakpoints increase with the slope.  A `BTreeMap` cannot be searched that
//! way on stable Rust, so the envelope is a plain vector in slope order
//! instead: the same one order, searched by slope or by breakpoint as needed,
//! and a neighbour is the next element rather than a tree walk.
//!
//! That trade is real and worth stating, because it is a trade.  `Vec::insert`
//! and `Vec::remove` shift every element past the position they touch, so the
//! build is `O(n)` amortized per key where a tree is `O(log n)`.  Measured
//! over the reference suite, the shifted tail is a fixed fraction of the
//! envelope, near `n / 33 000` — linear in `n`, not a constant.
//!
//!     tokens      envelope    mean shift    hull, vec    hull, tree    hull, C++
//!         59 089    44 588          1.4        1.13s         4.11s        1.32s
//!        178 226   178 225          7.7           --            --           --
//!      1 055 417  1 055 416        26.8       34.90s        81.84s       29.88s
//!
//! Two comparisons, and they do not agree.  Against the `BTreeMap` the array
//! wins at every size measured, by 3.6x at 59 089 tokens and still 2.3x at a
//! million: a shift is a memmove of contiguous lines, a tree step is a pointer
//! chased into a cold cache line, and the Rust tree pays for keeping a map and
//! its breakpoint index in step where one container would do.  Against the
//! C++ `std::multiset`, which is that one container, the array is 15% ahead at
//! 59 089 tokens and 15% behind at a million.  The linear term is real and it
//! does catch up.
//!
//! What it costs is not spread evenly.  Over the sudoku trace at a million
//! tokens, 97.6% of insertions shift fewer than four lines and 0.92% shift
//! more than a thousand — and that 0.92% moves 94% of all the elements ever
//! moved.  The mean of 26.8 is one rare deep insertion, not a tail that grows
//! under every key.  So the fix for the linear term is to bound the shift, not
//! to give up the array: the shape that makes it slow is also the shape that
//! makes it rare.

use crate::breakpoint::Break;
use crate::meta::HullMeta;
use core::cmp::Ordering;

/// A slope, ordered totally so that it can be searched for.
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
    pub m: Slope,
    pub b: f64,
    pub p: Break,
    pub meta: HullMeta,
}

/// The upper envelope of a set of lines, in increasing slope order.
///
/// Two invariants hold between calls, and both are what make the vector
/// searchable: the slopes are strictly increasing, and so are the
/// breakpoints.  The second is the reason a query can binary-search the same
/// array the slopes are keyed by.
#[derive(Default)]
pub struct Cht {
    lines: Vec<Line>,
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
    }

    /// The line at `i`, which is the handle a query and its neighbour walk use.
    pub fn get(&self, i: usize) -> &Line {
        &self.lines[i]
    }

    /// Set `x`'s breakpoint against `y`, and report whether `y` is now useless.
    ///
    /// `y = None` stands for the end of the envelope, where `x` runs forever.
    fn isect(&mut self, x: usize, y: Option<usize>) -> bool {
        let Some(yi) = y else {
            self.lines[x].p = Break::PosInf;
            return false;
        };
        let (my, by, yp) = {
            let l = &self.lines[yi];
            (l.m.get(), l.b, l.p)
        };
        let (mx, bx) = (self.lines[x].m.get(), self.lines[x].b);
        let p = Break::between(mx, bx, my, by);
        self.lines[x].p = p;
        p >= yp
    }

    /// The element after `i`, or `None` at the end of the envelope.
    fn after(&self, i: usize) -> Option<usize> {
        (i + 1 < self.lines.len()).then_some(i + 1)
    }

    /// Insert `y = m x + b` carrying `meta`, keeping only the upper envelope.
    pub fn add_line(&mut self, m: f64, b: f64, meta: HullMeta) {
        let s = Slope::new(m);
        let mut new_meta = meta;

        let at = match self.lines.binary_search_by(|l| l.m.cmp(&s)) {
            Ok(i) => {
                let l = &self.lines[i];
                if l.b == b {
                    let mut merged = l.meta;
                    merged.merge(&meta);
                    new_meta = merged;
                } else if l.b >= b {
                    return;
                }
                self.lines[i] = Line { m: s, b, p: Break::Unset, meta: new_meta };
                i
            }
            Err(i) => {
                self.lines.insert(i, Line { m: s, b, p: Break::Unset, meta: new_meta });
                i
            }
        };

        // Drop successors the new line has made redundant.
        while self.isect(at, self.after(at)) {
            self.lines.remove(at + 1);
        }

        // The predecessor may make the new line itself redundant.
        let mut x = at;
        if at > 0 {
            x = at - 1;
            if self.isect(x, Some(at)) {
                self.lines.remove(at);
                let after = (at < self.lines.len()).then_some(at);
                self.isect(x, after);
            }
        }

        // Walk left, dropping lines whose breakpoints are out of order.
        while x > 0 {
            let y = x;
            x = y - 1;
            if self.lines[x].p >= self.lines[y].p {
                self.lines.remove(y);
                let after = (y < self.lines.len()).then_some(y);
                self.isect(x, after);
            } else {
                break;
            }
        }
    }

    /// The line maximal at `x`: the first whose breakpoint reaches `x`.
    ///
    /// The breakpoints increase along the envelope, so this is a binary search
    /// of the same array the slopes are ordered by — which is what the second
    /// index existed for.
    pub fn argmax(&self, x: Break) -> Option<usize> {
        if self.lines.is_empty() {
            return None;
        }
        let i = self.lines.partition_point(|l| l.p < x);
        Some(i.min(self.lines.len() - 1))
    }

    /// The envelope in increasing slope order, for tests and for debugging.
    pub fn iter(&self) -> impl Iterator<Item = &Line> {
        self.lines.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The envelope's two invariants, which the binary searches rest on.
    fn well_formed(c: &Cht) {
        for w in c.lines.windows(2) {
            assert!(w[0].m < w[1].m, "slopes increase: {:?} {:?}", w[0].m, w[1].m);
            assert!(w[0].p < w[1].p, "breakpoints increase: {:?} {:?}", w[0].p, w[1].p);
        }
        if let Some(last) = c.lines.last() {
            assert_eq!(last.p, Break::PosInf, "the last line runs forever");
        }
    }

    /// `max_i (m_i x + b_i)`, scored directly.
    fn brute(lines: &[(f64, f64)], x: f64) -> f64 {
        lines.iter().map(|(m, b)| m * x + b).fold(f64::NEG_INFINITY, f64::max)
    }

    #[test]
    fn the_envelope_is_the_maximum_of_the_lines_it_was_given() {
        // xorshift64*, so the case list is reproducible without a dependency.
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state.wrapping_mul(0x2545_f491_4f6c_dd1d)
        };
        let mut coord = move || ((next() >> 40) as f64) - 8388608.0;

        for _ in 0..200 {
            let mut cht = Cht::new();
            let mut given = Vec::new();
            for _ in 0..60 {
                let (m, b) = (coord(), coord());
                cht.add_line(m, b, HullMeta::of([0.0, 0.0], 0));
                given.push((m, b));
                well_formed(&cht);

                for q in [-1e7, -1234.0, -1.0, 0.0, 1.0, 1234.0, 1e7] {
                    let i = cht.argmax(Break::ratio(q, 1.0)).expect("the envelope is not empty");
                    let l = cht.get(i);
                    assert_eq!(l.m.get() * q + l.b, brute(&given, q), "the maximum at {q}");
                }
            }
        }
    }

    #[test]
    fn a_line_under_the_envelope_is_dropped_and_an_equal_one_is_merged() {
        let mut cht = Cht::new();
        cht.add_line(0.0, 10.0, HullMeta::of([1.0, 0.0], 0));
        cht.add_line(1.0, 0.0, HullMeta::of([2.0, 0.0], 1));
        assert_eq!(cht.len(), 2);

        // Strictly below the envelope everywhere: the same slope, less offset.
        cht.add_line(0.0, 5.0, HullMeta::of([9.0, 0.0], 2));
        assert_eq!(cht.len(), 2);
        assert_eq!(cht.get(0).meta.count, 1, "the loser did not join the winner");

        // The same line twice: one node, two entries.
        cht.add_line(0.0, 10.0, HullMeta::of([3.0, 0.0], 3));
        assert_eq!(cht.len(), 2);
        assert_eq!(cht.get(0).meta.count, 2);
        assert_eq!(cht.get(0).meta.vsum[0], 4.0);
    }

    #[test]
    fn minus_zero_and_zero_are_one_line() {
        let mut cht = Cht::new();
        cht.add_line(0.0, 1.0, HullMeta::of([1.0, 0.0], 0));
        cht.add_line(-0.0, 1.0, HullMeta::of([1.0, 0.0], 1));
        assert_eq!(cht.len(), 1);
        assert_eq!(cht.get(0).meta.count, 2);
    }
}

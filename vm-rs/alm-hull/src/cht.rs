//! The dynamic convex hull trick: an upper envelope of lines under insertion.
//!
//! Ported from `_HullCHT` in `attention/hull2d_cht.h`.  The C++ keeps the
//! envelope in one `std::multiset` ordered by slope and searches it by
//! breakpoint through a heterogeneous comparator, which works because the
//! breakpoints increase with the slope.  Here the envelope is a plain vector
//! in slope order: the same one order, searched by slope or by breakpoint as
//! needed, and a neighbour is the next element rather than a tree walk.
//!
//! The multiset is expressible, and was measured rather than assumed away.  A
//! `BTreeSet` of an enum that is either a line or a query, ordered by slope
//! between two lines and by the line's breakpoint against a query, is the C++
//! comparator exactly; stable Rust has no cursor API (`btree_cursors`,
//! rust#107540), but a `range` iterator is one descent and then O(1) per step,
//! and removals defer out of the walk.  Built that way it agrees with this
//! vector line for line.  What it costs is the arrival order the traces
//! actually have: inserting 262 144 parabolic keys in position order takes it
//! 0.216s against this container's 0.041s.  It is the faster container only
//! where this one is pathological -- 0.106s against 92.4s in the reverse
//! order -- which is `alm-stress`, not a trace.
//!
//! That trade is real and worth stating, because it is a trade.  Insertion
//! shifts every line past the point it touches, so the build is `O(n)`
//! amortized per key where a tree is `O(log n)`, and the shifted tail is not
//! a constant: measured over the reference suite it is a fixed fraction of
//! the envelope, near `n / 33 000`.
//!
//! Back to back on one machine, the `hull` bucket alone:
//!
//! ```text
//! tokens      envelope    mean shift      vec    BTreeMap      C++    vec/C++
//!     59 089    44 588          1.4     1.07s       4.11s    1.32s      0.81
//!    178 226   178 225          7.7     3.70s          --    4.55s      0.81
//!  1 055 417  1 055 416        26.8    26.03s      81.84s   29.64s      0.88
//! ```
//!
//! The constant is what decides it, and the constant is large.  A shift is a
//! memmove of contiguous lines; a tree step is a pointer chased into a cold
//! cache line, and the C++ container is searched twice per key.  The Rust
//! `BTreeMap` was worse than either, because it needed a second index in step
//! with the map where one container serves here.
//!
//! The linear term does close, but slowly.  Fitting the three sizes above
//! puts the array level with the C++ multiset near `1.4e8` tokens — the same
//! order as the `9.5e7` at which the score `2qk - k^2` leaves the exact
//! integers (`grid.rs`) and this machine stops answering correctly at all.
//! The array is the right shape for every `n` this engine can be trusted at,
//! and only for those.
//!
//! What the linear term costs is not spread evenly, which is what leaves room
//! to shrink it further.  Over the sudoku trace 97.6% of insertions shift
//! fewer than four lines and 0.92% shift more than a thousand — and that
//! 0.92% moves 94% of everything ever moved.  The mean of 26.8 is one rare
//! deep insertion, not a tail that grows under every key, so bounding the
//! shift would not cost the locality the array is kept for.

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

    /// The breakpoint of the line at `i` against `r`, or `PosInf` past the end.
    fn break_against(&self, i: usize, r: Option<Line>) -> Break {
        let x = &self.lines[i];
        match r {
            Some(y) => Break::between(x.m.get(), x.b, y.m.get(), y.b),
            None => Break::PosInf,
        }
    }

    /// Insert `y = m x + b` carrying `meta`, keeping only the upper envelope.
    ///
    /// The erase loops decide what survives by reading the array, never by
    /// editing it: an insertion that drops `k` neighbours rewrites the tail
    /// once rather than `k + 1` times.  That is not a constant factor to
    /// shrug at.  Over the sudoku trace the container moved 4.2e9 lines of
    /// 80 bytes each, and 0.8 of every insertion's moves was a `remove`
    /// walking the same tail the `insert` had just walked.
    ///
    /// What survives is `lines[..lo] ++ [new] ++ lines[hi..]`, with `lo` and
    /// `hi` closing in from the insertion point as neighbours fall, so the
    /// whole of it is one `copy_within`.
    pub fn add_line(&mut self, m: f64, b: f64, meta: HullMeta) {
        let s = Slope::new(m);
        let mut new_meta = meta;

        // Where the new line goes, and where the tail it does not touch begins.
        let (at, mut hi) = match self.lines.binary_search_by(|l| l.m.cmp(&s)) {
            Ok(i) => {
                let l = &self.lines[i];
                if l.b == b {
                    let mut merged = l.meta;
                    merged.merge(&meta);
                    new_meta = merged;
                } else if l.b >= b {
                    return;
                }
                (i, i + 1)
            }
            Err(i) => (i, i),
        };
        let mut new = Line { m: s, b, p: Break::Unset, meta: new_meta };

        // Drop successors the new line has made redundant.
        loop {
            let Some(succ) = self.lines.get(hi) else {
                new.p = Break::PosInf;
                break;
            };
            new.p = Break::between(new.m.get(), new.b, succ.m.get(), succ.b);
            if new.p >= succ.p {
                hi += 1;
            } else {
                break;
            }
        }

        // The predecessor may make the new line itself redundant, and then
        // walk left dropping lines whose breakpoints are out of order.
        let mut keep = true;
        let mut lo = at;
        if at > 0 {
            let mut cur = at - 1;
            let mut p = self.break_against(cur, Some(new));
            if p >= new.p {
                keep = false;
                p = self.break_against(cur, self.lines.get(hi).copied());
            }
            let right = if keep { Some(new) } else { self.lines.get(hi).copied() };
            while cur > 0 && self.lines[cur - 1].p >= p {
                cur -= 1;
                p = self.break_against(cur, right);
            }
            self.lines[cur].p = p;
            lo = cur + 1;
        }

        // One move for the whole of it.
        let len = self.lines.len();
        let dest = lo + usize::from(keep);
        if dest > hi {
            // Nothing was dropped, so this is a plain insertion.
            self.lines.insert(lo, new);
            return;
        }
        if dest < hi {
            self.lines.copy_within(hi..len, dest);
            self.lines.truncate(len - (hi - dest));
        }
        if keep {
            self.lines[lo] = new;
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

//! The dynamic convex hull trick in a vector: the envelope the tree is tested
//! against.
//!
//! This was the engine's container and is not any more, which is worth
//! stating in one place because the reason is not the one the measurements
//! pointed at.  `HullBuild.lean` charges a build two searches per key and one
//! unit per erase and proves it `O(n log n)` from that tariff; the tariff is
//! only true of a container whose erase at a known cursor is amortized
//! constant.  Erasing from a vector costs the tail it moves, so `buildCost_le`
//! was a theorem about a container the engine did not have.  `envelope.rs` is
//! the one it describes.
//!
//! What the vector was, it was honestly: the faster container on every trace
//! this engine can be trusted at.  Back to back on one machine, the `hull`
//! bucket alone:
//!
//! ```text
//! tokens      envelope    mean shift      vec       tree    BTreeMap      C++
//!     59 089    44 588          1.4     1.05s      1.20s       4.11s    1.32s
//!    178 226   178 225          7.7     3.78s      4.05s          --    4.55s
//!  1 055 417  1 055 416        26.8    26.11s     27.07s      81.84s   29.64s
//! ```
//!
//! Each figure is the better of two runs taken alternately; the vector's
//! spread across runs (3.78s and 4.26s on the middle trace) is wider than the
//! gap it wins by there.  Four to thirteen per cent, which is what an
//! insertion that shifts nothing on a trace whose keys arrive nearly sorted
//! is worth.  The same insertion off that
//! order is quadratic: 262 144 keys in descending order build in 102.9s
//! against the tree's 0.080s, and `alm-stress` will show it in any of five
//! orders.  Nothing in the engine promises the sorted one -- `cache.rs` reads
//! the keys from a learned projection -- so the vector was one trace away
//! from stopping.
//!
//! It is kept because two containers that must agree are worth more than one
//! that cannot be checked: `envelope.rs` is tested against this line for
//! line, over random lines, over the parabolic lift, over every arrival order
//! and over repeated slopes, the same way `BruteAttentionHead` is kept to
//! test `HardAttentionHead`.
//!
//! What the shift cost when it was paid is still the clearest picture of why
//! it had to go.  Over the sudoku trace 97.6% of insertions shifted fewer
//! than four lines and 0.92% shifted more than a thousand -- and that 0.92%
//! moved 94% of the 4.2e9 lines of 80 bytes the container ever moved.  The
//! mean of 26.8 was one rare deep insertion, not a tail that grows under
//! every key.

use crate::breakpoint::Break;
use crate::meta::HullMeta;
pub use crate::tree::{Line, Slope};

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

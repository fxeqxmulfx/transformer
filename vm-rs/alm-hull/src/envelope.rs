//! The dynamic convex hull trick: an upper envelope of lines under insertion.
//!
//! Ported from `_HullCHT` in `attention/hull2d_cht.h`.  The C++ keeps the
//! envelope in one `std::multiset` ordered by slope and searched by breakpoint
//! through a heterogeneous comparator, which works because the breakpoints
//! increase with the slope.  This keeps it in `tree.rs`, which is that
//! container with cursors: one order, searched by slope or by breakpoint as
//! needed, and a neighbour is a step rather than a search.
//!
//! What the container has to give, `HullBuild.lean` says exactly.  `buildCost`
//! charges a build two searches per key and one unit for each erase, and
//! `buildCost_le` proves it `O(n log n)` from that -- a tariff that holds only
//! where erasing at a known cursor is amortized constant.  A vector cannot
//! meet it: its erase costs the tail it moves.  This port is cheaper than the
//! tariff, at one search per key, because `insert_before` takes the place
//! `lower_bound_slope` already found.
//!
//! So the erase loops here read and unlink in the same walk.  The lines a new
//! one hides are dropped at the cursor that found them, nothing else moves,
//! and the new line is linked where the walk stopped.  None of that depends
//! on the order the keys arrive in, which is the point: `alm-stress` builds
//! 262 144 keys in each of five orders and the spread is 0.030s to 0.123s.
//!
//! Two tariffs account for that spread, and they disagree about it.
//! `BuildOrder.lean` prices the container the C++ header uses -- two searches
//! per key, because `std::map::insert` after a `lower_bound` searches again --
//! and can account for none of the spread: `buildPrices_spread` caps the
//! difference between two orders at one erase per key, under three per cent at
//! this size (`stress_spread_le`), and on lifted keys no erase fires at all,
//! so `buildPrices_paraboloid` charges every order one single number.
//!
//! `BuildFinger.lean` prices what this file runs, which is less.  `add_line`
//! descends once, not twice: `insert_before` is handed the cursor
//! `lower_bound_slope` returned, and every erase after it is a cursor walk.
//! And on a key whose slope falls outside the span the envelope covers there
//! is no descent at all, because `tree.rs` holds both ends in a field --
//! `portCost_le_buildCost` is that saving, and `portCost_of_all_ends` is the
//! near-sorted build, linear with no logarithm in it.  That tariff does see
//! part of the spread: `portCost_moves_on_the_paraboloid` gives two orders of
//! the same lifted keys a factor of `log n` apart, and `alm-stress` counts the
//! descents and finds three of its five orders have none.  The remainder is
//! memory: two of the orders descend on the same keys to within fifty and
//! still differ by half again, which is the whole reason this is `tree.rs` and
//! not a vector.
//!
//! One thing the tariff never charged for, the port used to pay anyway.  Both
//! walks here probe past the end -- the successor loop reads `next(hi)` and
//! the predecessor walk reads `prev(at)` -- and finding nothing there cost a
//! climb to the root until `tree.rs` answered those two from `ends`.  The
//! model charged zero for a probe that fails; now so does the code.

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

    /// The leftmost line, or `NIL` when the envelope is empty.
    ///
    /// Held rather than walked to, and so is `last`.  They are what decides
    /// whether an insertion descends at all: `lower_bound_slope` answers from
    /// them when the new slope is outside the range they span.
    pub fn first(&self) -> u32 {
        self.t.first()
    }

    /// The rightmost line, or `NIL` when the envelope is empty.
    pub fn last(&self) -> u32 {
        self.t.last()
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

    /// The envelope's two invariants, which both searches rest on.
    fn well_formed(e: &Envelope) {
        let lines: Vec<Line> = e.iter().collect();
        for w in lines.windows(2) {
            assert!(w[0].m < w[1].m, "slopes increase: {:?} {:?}", w[0].m, w[1].m);
            assert!(w[0].p < w[1].p, "breakpoints increase: {:?} {:?}", w[0].p, w[1].p);
        }
        if let Some(l) = lines.last() {
            assert_eq!(l.p, Break::PosInf, "the last line runs forever");
        }
    }

    /// `max_i (m_i x + b_i)`, scored directly over everything ever inserted.
    fn brute(given: &[(f64, f64)], x: f64) -> f64 {
        given.iter().map(|(m, b)| m * x + b).fold(f64::NEG_INFINITY, f64::max)
    }

    /// The envelope answers each `x` with the maximum of the lines it was given.
    ///
    /// This is the reference, and it is not another envelope: whatever the
    /// container does, the line it hands back must score what the whole set
    /// scores at that point.
    fn answers_the_maximum(e: &Envelope, given: &[(f64, f64)], xs: &[f64]) {
        for &q in xs {
            let i = e.argmax(Break::ratio(q, 1.0)).expect("the envelope is not empty");
            let l = e.get(i);
            assert_eq!(l.m.get() * q + l.b, brute(given, q), "the maximum at {q}");
        }
    }

    /// xorshift64*, so every case list is reproducible without a dependency.
    fn rng() -> impl FnMut() -> u64 {
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        }
    }

    const PROBES: [f64; 7] = [-1e7, -1234.0, -1.0, 0.0, 1.0, 1234.0, 1e7];

    #[test]
    fn the_envelope_is_the_maximum_of_the_lines_it_was_given() {
        let mut next = rng();
        let mut coord = move || ((next() >> 40) as f64) - 8388608.0;
        for _ in 0..200 {
            let mut e = Envelope::new();
            let mut given = Vec::new();
            for _ in 0..60 {
                let (m, b) = (coord(), coord());
                e.add_line(m, b, HullMeta::of([0.0, 0.0], 0));
                given.push((m, b));
                well_formed(&e);
                answers_the_maximum(&e, &given, &PROBES);
            }
        }
    }

    /// The parabolic lift, which is the only shape the engine ever inserts.
    ///
    /// Strictly concave in the slope, so no key is ever dropped and the
    /// envelope is every key that arrived.
    #[test]
    fn the_lifted_keys_all_survive_and_answer() {
        let mut next = rng();
        for &span in &[16u64, 1024, 65536] {
            let mut e = Envelope::new();
            let mut given = Vec::new();
            let mut seen = std::collections::BTreeSet::new();
            for _ in 0..500 {
                let k = next() % span;
                let (m, b) = (2.0 * k as f64, -((k * k) as f64));
                e.add_line(m, b, HullMeta::of([0.0, 0.0], 0));
                given.push((m, b));
                seen.insert(k);
            }
            assert_eq!(e.len(), seen.len(), "a lifted key was dropped at span {span}");
            well_formed(&e);
            let xs: Vec<f64> = (0..64).map(|i| f64::from(i) * (span as f64) / 64.0).collect();
            answers_the_maximum(&e, &given, &xs);
        }
    }

    /// Every arrival order builds the same envelope and answers the same way.
    ///
    /// This is what the container is kept for: the five orders `alm-stress`
    /// drives differ only in the path to the answer, and the answer must not
    /// notice.
    ///
    /// The envelope also keeps every line, in every order, which is the
    /// hypothesis both build tariffs need: no erase fires on lifted keys, so
    /// no order can be charged for one.  It is all
    /// `ALM.BuildOrder.buildPrices_paraboloid` is left with, and it is why
    /// `ALM.BuildFinger.portCost_moves_on_the_paraboloid` can put the whole
    /// remaining difference between two orders on the searches.
    #[test]
    fn every_arrival_order_builds_the_same_envelope() {
        let mut want: Option<Vec<(f64, f64)>> = None;
        for &step in &[1i64, -1, 7, -7, 2003] {
            let mut e = Envelope::new();
            let mut given = Vec::new();
            // 601 is prime, so every step below walks the same 601 keys.
            for i in 0..601i64 {
                let k = (i * step).rem_euclid(601) as f64;
                let (m, b) = (2.0 * k, -k * k);
                e.add_line(m, b, HullMeta::of([0.0, 0.0], 0));
                given.push((m, b));
            }
            well_formed(&e);
            let xs: Vec<f64> = (0..64).map(|i| f64::from(i) * 20.0).collect();
            answers_the_maximum(&e, &given, &xs);
            let got: Vec<(f64, f64)> = e.iter().map(|l| (l.m.get(), l.b)).collect();
            assert_eq!(got.len(), 601, "order {step} erased a lifted key");
            match &want {
                None => want = Some(got),
                Some(w) => {
                    assert_eq!(got.len(), w.len(), "order {step} built {} lines, not {}", got.len(), w.len());
                    for (i, (g, w)) in got.iter().zip(w).enumerate() {
                        assert_eq!(g, w, "order {step} differs at line {i}");
                    }
                }
            }
        }
    }

    /// Repeated slopes, the branch that merges or drops rather than inserting.
    #[test]
    fn repeated_slopes_still_answer_the_maximum() {
        let mut next = rng();
        let mut e = Envelope::new();
        let mut given = Vec::new();
        for _ in 0..800 {
            let m = (next() % 7) as f64;
            let b = (next() % 11) as f64 - 5.0;
            e.add_line(m, b, HullMeta::of([0.0, 0.0], 0));
            given.push((m, b));
            answers_the_maximum(&e, &given, &PROBES);
        }
        well_formed(&e);
    }

    #[test]
    fn a_line_under_the_envelope_is_dropped_and_an_equal_one_is_merged() {
        let mut e = Envelope::new();
        e.add_line(0.0, 10.0, HullMeta::of([1.0, 0.0], 0));
        e.add_line(1.0, 0.0, HullMeta::of([2.0, 0.0], 1));
        assert_eq!(e.len(), 2);

        // Strictly below the envelope everywhere: the same slope, less offset.
        e.add_line(0.0, 5.0, HullMeta::of([9.0, 0.0], 2));
        assert_eq!(e.len(), 2);
        let first = e.iter().next().expect("the envelope is not empty");
        assert_eq!(first.meta.count, 1, "the loser did not join the winner");

        // The same line twice: one node, two entries.
        e.add_line(0.0, 10.0, HullMeta::of([3.0, 0.0], 3));
        assert_eq!(e.len(), 2);
        let first = e.iter().next().expect("the envelope is not empty");
        assert_eq!(first.meta.count, 2);
        assert_eq!(first.meta.vsum[0], 4.0);
    }

    #[test]
    fn minus_zero_and_zero_are_one_line() {
        let mut e = Envelope::new();
        e.add_line(0.0, 1.0, HullMeta::of([1.0, 0.0], 0));
        e.add_line(-0.0, 1.0, HullMeta::of([1.0, 0.0], 1));
        assert_eq!(e.len(), 1);
        let first = e.iter().next().expect("the envelope is not empty");
        assert_eq!(first.meta.count, 2);
    }
}

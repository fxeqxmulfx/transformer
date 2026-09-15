//! The two hard-attention heads: the hull one, and the brute-force reference.
//!
//! Ported from `HullHalf`, `HardAttentionHead` and `BruteAttentionHead` in
//! `attention/hull2d_cht.h`.  The pair exists to be differentially tested
//! against each other: the C++ release disagrees with its own reference on ties
//! (`todo3.md` section 3), and reproducing both is how that is measured here.

use core::cell::Cell;

use crate::breakpoint::Break;
use crate::gap::ScoreGaps;
use crate::envelope::Envelope;
use crate::grid::GridWitness;
use crate::meta::{HullMeta, TieBreak};
use crate::tree::NIL;

/// One envelope: the upper hull maximises `kx * m + ky`, the lower minimises it
/// by maximising the negated line.
pub struct HullHalf {
    cht: Envelope,
    is_upper: bool,
}

/// What a query found: the resolved payload, the winning score and key.
#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub out: [f64; 2],
    pub score: f64,
    pub best_kx: f64,
    /// The winning key, as the head stored it.
    pub best_key: [f64; 2],
    /// The entries at it, before the tie-break collapsed them into `out`.
    /// A caller holding a second container has to merge, not resolve.
    pub meta: HullMeta,
}

impl HullHalf {
    pub fn new(is_upper: bool) -> HullHalf {
        HullHalf { cht: Envelope::new(), is_upper }
    }

    pub fn len(&self) -> usize {
        self.cht.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cht.is_empty()
    }

    pub fn clear(&mut self) {
        self.cht.clear();
    }

    pub fn insert(&mut self, kx: f64, ky: f64, val: [f64; 2], seq: i32) {
        let meta = HullMeta::of(val, seq);
        if self.is_upper {
            self.cht.add_line(kx, ky, meta);
        } else {
            self.cht.add_line(-kx, -ky, meta);
        }
    }

    /// The stored key of the line at `i`, undoing the negation of a lower hull.
    fn key_at(&self, i: u32) -> (f64, f64) {
        let (m, b) = self.cht.key(i);
        if self.is_upper {
            (m, b)
        } else {
            (-m, -b)
        }
    }

    pub fn query(&self, qx: f64, qy: f64, tb: TieBreak) -> Option<Hit> {
        if self.cht.is_empty() {
            return None;
        }

        let at = if qy == 0.0 {
            if qx >= 0.0 {
                Break::PosInf
            } else {
                Break::NegInf
            }
        } else {
            Break::ratio(qx, qy)
        };
        let best = self.cht.argmax(at)?;

        let (kx_best, ky_best) = self.key_at(best);
        let best_score = qx * kx_best + qy * ky_best;

        // A query with `qy == 0` reads one extreme of the envelope and stops:
        // the ties there are already collapsed into that node's aggregate.
        if qy == 0.0 {
            let meta = self.cht.meta_of(best);
            let out = meta.resolve(tb);
            let best_key = [kx_best, ky_best];
            return Some(Hit { out, score: best_score, best_kx: kx_best, best_key, meta });
        }

        let mut combined = HullMeta::default();
        combined.merge(&self.cht.meta_of(best));

        // The ties either side of the winner, walked with the cursor that
        // found it: a neighbour is one step, not another search.
        let mut left = self.cht.prev(best);
        while left != NIL {
            let (kx, ky) = self.key_at(left);
            if qx * kx + qy * ky != best_score {
                break;
            }
            combined.merge(&self.cht.meta_of(left));
            left = self.cht.prev(left);
        }

        let mut right = self.cht.next(best);
        while right != NIL {
            let (kx, ky) = self.key_at(right);
            if qx * kx + qy * ky != best_score {
                break;
            }
            combined.merge(&self.cht.meta_of(right));
            right = self.cht.next(right);
        }

        let out = combined.resolve(tb);
        let best_key = [kx_best, ky_best];
        Some(Hit { out, score: best_score, best_kx: kx_best, best_key, meta: combined })
    }
}

/// The `O(log n)` hard-attention head: two envelopes plus the degenerate cases.
pub struct HardAttentionHead {
    upper: HullHalf,
    lower: HullHalf,
    global: HullMeta,
    left_meta: HullMeta,
    right_meta: HullMeta,
    min_kx: f64,
    max_kx: f64,
    n: usize,
    /// Written from `query`, which takes `&self`: the witness is an
    /// observation of the head, not part of its answer.
    grid: Cell<GridWitness>,
}

impl Default for HardAttentionHead {
    fn default() -> Self {
        HardAttentionHead {
            upper: HullHalf::new(true),
            lower: HullHalf::new(false),
            global: HullMeta::default(),
            left_meta: HullMeta::default(),
            right_meta: HullMeta::default(),
            min_kx: f64::INFINITY,
            max_kx: f64::NEG_INFINITY,
            n: 0,
            grid: Cell::new(GridWitness::default()),
        }
    }
}

impl HardAttentionHead {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    pub fn clear(&mut self) {
        *self = Self::default();
    }

    pub fn insert(&mut self, key: [f64; 2], val: [f64; 2], seq: i32) {
        self.global.add(val, seq);

        if key[0] < self.min_kx {
            self.min_kx = key[0];
            self.left_meta = HullMeta::default();
        }
        if key[0] == self.min_kx {
            self.left_meta.add(val, seq);
        }

        if key[0] > self.max_kx {
            self.max_kx = key[0];
            self.right_meta = HullMeta::default();
        }
        if key[0] == self.max_kx {
            self.right_meta.add(val, seq);
        }

        self.upper.insert(key[0], key[1], val, seq);
        self.lower.insert(key[0], key[1], val, seq);
        self.n += 1;
    }

    /// What this head has answered that float64 could not separate — empty
    /// unless a *winning score* has crossed `2^53`.  See `grid`.
    pub fn grid_witness(&self) -> GridWitness {
        self.grid.get()
    }

    /// The margin a winner must beat: for the parabolic keys of this machine
    /// the runner-up's true score is at least one integer step away, and one
    /// step in the score is `|qy|`.  The `qy == 0` shortcut compares along the
    /// other axis, where a step is `|qx|`.
    fn note(&self, score: f64, margin: f64, query: [f64; 2], key: Option<[f64; 2]>) {
        let mut w = self.grid.get();
        w.observe(score, margin, query, key);
        self.grid.set(w);
    }

    pub fn query(&self, q: [f64; 2], tb: TieBreak) -> Option<[f64; 2]> {
        let (qx, qy) = (q[0], q[1]);
        if qy == 0.0 {
            if self.n == 0 {
                return None;
            }
            return Some(if qx > 0.0 {
                self.note(qx * self.max_kx, qx, q, None);
                self.right_meta.resolve(tb)
            } else if qx < 0.0 {
                self.note(qx * self.min_kx, qx, q, None);
                self.left_meta.resolve(tb)
            } else {
                self.global.resolve(tb)
            });
        }
        self.hit(q, tb).map(|h| {
            self.note(h.score, qy, q, Some(h.best_key));
            h.out
        })
    }

    /// The same query, handing back the winner itself rather than only what it
    /// resolved to.
    ///
    /// `lifthead.rs` needs it: that head holds the cleared entries in a head of
    /// its own, and on the queries where the clear marker gives it no margin it
    /// has to compare this head's winner against its own and merge what ties.
    /// A resolved payload can do neither.
    ///
    /// The axis is not answered here.  At `qy == 0` the ordinate is multiplied
    /// away, there is no single winning key to hand back, and the caller that
    /// needs this holds its own ends for exactly that case.
    ///
    /// Nor is the grid witness written here, and that is the other half of why
    /// this is separate from `query`: a caller comparing this head's winner
    /// against another container's has not decided anything yet, and a winner
    /// that goes on to lose is not a query answered with no margin to spare.
    /// `query` notes what it returns; a caller of `hit` notes what it keeps.
    pub fn hit(&self, q: [f64; 2], tb: TieBreak) -> Option<Hit> {
        let (qx, qy) = (q[0], q[1]);
        if qy == 0.0 {
            return None;
        }
        let half = if qy > 0.0 { &self.upper } else { &self.lower };
        half.query(qx, qy, tb)
    }
}

/// The `O(n)` reference head: score everything, keep the maxima.
#[derive(Default)]
pub struct BruteAttentionHead {
    entries: Vec<([f64; 2], [f64; 2], i32)>,
    grid: Cell<GridWitness>,
    gaps: Cell<ScoreGaps>,
}

impl BruteAttentionHead {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.grid.set(GridWitness::default());
        self.gaps.set(ScoreGaps::default());
    }

    /// How close the runner-up came, in key steps — the diagnostic of
    /// `todo3.md` section 2a.  Only this head can report it: it scores every
    /// entry anyway, while the hull visits the winner and its ties and stops.
    pub fn gap_witness(&self) -> ScoreGaps {
        self.gaps.get()
    }

    pub fn insert(&mut self, key: [f64; 2], val: [f64; 2], seq: i32) {
        self.entries.push((key, val, seq));
    }

    /// As `HardAttentionHead::grid_witness`: the wall is a property of the
    /// arithmetic, so the brute head crosses it on exactly the same query.
    pub fn grid_witness(&self) -> GridWitness {
        self.grid.get()
    }

    pub fn query(&self, q: [f64; 2], tb: TieBreak) -> Option<[f64; 2]> {
        if self.entries.is_empty() {
            return None;
        }
        let score = |k: &[f64; 2]| q[0] * k[0] + q[1] * k[1];
        let max = self.entries.iter().map(|(k, _, _)| score(k)).fold(f64::NEG_INFINITY, f64::max);
        let winner = self.entries.iter().find(|(k, _, _)| score(k) == max).map(|(k, _, _)| *k);
        let mut w = self.grid.get();
        w.observe(max, if q[1] != 0.0 { q[1] } else { q[0] }, q, winner);
        self.grid.set(w);
        let second = self
            .entries
            .iter()
            .map(|(k, _, _)| score(k))
            .filter(|&s| s < max)
            .fold(f64::NEG_INFINITY, f64::max);
        // The size the rounding of `q0*k0 + q1*k1` is relative to, taken over
        // the two entries actually in the race: an error bound drawn from a
        // key that lost by miles would say nothing about this comparison.
        let terms = |k: &[f64; 2]| (q[0] * k[0]).abs() + (q[1] * k[1]).abs();
        let worst_terms = self
            .entries
            .iter()
            .filter(|(k, _, _)| score(k) == max || score(k) == second)
            .map(|(k, _, _)| terms(k))
            .fold(0.0f64, f64::max);
        let mut g = self.gaps.get();
        // Where the bound does not decide the comparison, ask arithmetic that
        // does.  `ALM.DotError.cmp_of_dot_guard` licenses the float answer
        // only when the gap clears twice the bound; on the handful of queries
        // where it does not, `exact::dot_cmp` says whether the doubt was
        // warranted.  It is a handful, so the cost is nothing.
        if g.observe(max, second, q, worst_terms) {
            // Which key wins under exact arithmetic, and is it one of the
            // keys the float scan called maximal?  If not, no tie-break can
            // recover it: the head answered with a key that does not win.
            let mut best = self.entries[0].0;
            for (k, _, _) in &self.entries {
                if crate::exact::dot_cmp(q, *k, best) == core::cmp::Ordering::Greater {
                    best = *k;
                }
            }
            if score(&best) != max {
                g.observe_misranked();
            }
        }
        self.gaps.set(g);

        let mut meta = HullMeta::default();
        for (k, v, seq) in &self.entries {
            if score(k) == max {
                meta.add(*v, *seq);
            }
        }
        Some(meta.resolve(tb))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_query_the_bound_doubts_and_the_exact_dot_product_overturns() {
        // Three roundings are not monotone in the exact score, so the float
        // scan really can put the wrong key first -- one rounding could only
        // ever tie.  Here `7 * k0` rounds up for the first key by more than
        // the quarter that separates the two exact scores, so the float order
        // is the reverse of the exact one.  `ALM.DotError.cmp_of_dot_guard`
        // is exactly what fails: the gap is 2 and twice the bound is 5.8.
        let q = [7.0, 1.0];
        let mut h = BruteAttentionHead::new();
        h.insert([1725231710801247.5, -1016251981013381.0], [1.0, 0.0], 0);
        h.insert([1725231710801247.0, -1016251981013377.2], [2.0, 0.0], 1);
        assert_eq!(h.query(q, TieBreak::Latest), Some([1.0, 0.0]), "float picks the first");
        let g = h.gap_witness();
        assert_eq!(g.unresolved, 1, "and the bound says so");
        assert_eq!(g.misranked, 1, "and it was right to");
        assert!(g.worst_guard > 2.9 && g.worst_guard < 2.91, "guard: {}", g.worst_guard);
    }

    #[test]
    fn a_query_the_bound_doubts_and_the_exact_dot_product_upholds() {
        // The same machinery on keys a whole step apart: the bound is loose
        // enough to doubt a gap it should not, and the exact comparison sends
        // it away.  This is the shape of all 61 unresolved queries on
        // `fibonacci`, where `misranked` came back zero.
        let q = [2.0e18, 1.0];
        let mut h = BruteAttentionHead::new();
        h.insert([1.0, 0.0], [1.0, 0.0], 0);
        h.insert([1.0 + f64::EPSILON, 0.0], [2.0, 0.0], 1);
        assert_eq!(h.query(q, TieBreak::Latest), Some([2.0, 0.0]));
        let g = h.gap_witness();
        assert_eq!(g.unresolved, 1, "the gap is one ulp of the product");
        assert_eq!(g.misranked, 0, "but the float order is the exact one");
    }
}

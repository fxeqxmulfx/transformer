//! How close two keys of one head ever get.
//!
//! `alm-hull/src/lift.rs` says which family a key belongs to; this says
//! whether the live family is spread out enough for the theorems about it to
//! apply.  `ALM.HullNear` answers a query with the nearest key and pays for it
//! with integrality (`sq_dist_gap_of_near_int` uses `z != z'` only to get
//! `1 <= |z' - z|`), and the shipped keys are not integers — `lift.rs` counts
//! `576` of `137389` of them off the grid on `hello`.  `ALM.HullSep` replaces
//! the unit by a measurement, and this is the measurement.
//!
//! Two numbers come out of `ALM.HullSep`:
//!
//! * `sq_dist_gap_of_sep` — keys `sep` apart give each key a window of
//!   `(sep^2 - A) / (2*sep)` around it in which it alone answers.  That is
//!   `ALM.HullNear`'s `(1 - A) / 2` at `sep = 1`, and it closes at
//!   `sep^2 = A`.  `window` below is that expression.
//! * `the_shipped_separation_floor` — for the shipped `A = MARK_SPREAD` the
//!   floor `sqrt(A)` is between `0.657` and `0.659`.  Below it the
//!   nearest-key rule is not unproved but false: `nearest_fails_of_close`
//!   exhibits two keys `d` apart with `d^2 < A` where the further one wins at
//!   the nearer one's own abscissa, whatever the container is implemented
//!   with.  `SEP_FLOOR` is that number and `under_floor` is the count.
//!
//! The minimum over all pairs is the minimum over adjacent pairs of the
//! sorted keys, so inserting into an ordered set and looking at the two
//! neighbours is enough: a pair stops being adjacent only after both its
//! neighbours' gaps have already been recorded, and those are smaller.

use std::collections::BTreeSet;

use crate::lift::MARK_SPREAD;

/// `sqrt(MARK_SPREAD)`: the separation a live head's keys must clear for the
/// nearest key to be the answer.  `ALM.HullSep.the_shipped_separation_floor`
/// brackets it between `0.657` and `0.659`.
pub const SEP_FLOOR: f64 = 0.6578818376172799;

/// The window `(sep^2 - A) / (2*sep)` of `ALM.HullSep.sq_dist_gap_of_sep`:
/// how far a query may stray from a key and still be answered by it alone,
/// when the nearest other key is `sep` away and the offsets spread by `A`.
/// `None` below the floor, where there is no such window.
pub fn window(sep: f64, spread: f64) -> Option<f64> {
    if !(sep > 0.0) || sep * sep <= spread {
        return None;
    }
    Some((sep * sep - spread) / (2.0 * sep))
}

/// A total order on the finite doubles, so a `BTreeSet` can hold keys.
fn order_bits(x: f64) -> u64 {
    let b = x.to_bits();
    if b & (1 << 63) != 0 {
        !b
    } else {
        b | (1 << 63)
    }
}

fn from_order_bits(u: u64) -> f64 {
    f64::from_bits(if u & (1 << 63) != 0 { u & !(1 << 63) } else { !u })
}

/// The closest two keys of one head ever came.
#[derive(Clone, Default, Debug, PartialEq)]
pub struct SepWitness {
    keys: BTreeSet<u64>,
    /// Keys offered, including repeats of one already held.
    pub total: usize,
    /// Repeats: a key equal to one already in the head.  These are not a
    /// separation failure — `ALM.HullMark.Marked.onePerKey` is about one line
    /// per *distinct* key, and a rewrite of the same key is what the recency
    /// offset exists to order.
    pub repeats: usize,
    /// Distinct keys held.
    pub distinct: usize,
    /// Insertions whose nearest neighbour was closer than `SEP_FLOOR`.
    pub under_floor: usize,
    /// The smallest gap between two distinct keys, `INFINITY` until there are
    /// two of them.
    pub worst: f64,
    /// The pair that realised it.
    pub worst_at: Option<[f64; 2]>,
}

impl SepWitness {
    pub fn new() -> SepWitness {
        SepWitness { worst: f64::INFINITY, ..SepWitness::default() }
    }

    /// Offer one key abscissa — `kx / 2`, the `k` of `ALM.HullMark.markKey`.
    /// Non-finite keys are counted and dropped: they have no separation, and
    /// `lift.rs` already reports them as off every family.
    pub fn observe(&mut self, k: f64) {
        self.total += 1;
        if !k.is_finite() {
            return;
        }
        let u = order_bits(k);
        if !self.keys.insert(u) {
            self.repeats += 1;
            return;
        }
        self.distinct += 1;
        let below = self.keys.range(..u).next_back().map(|&v| k - from_order_bits(v));
        let above = self.keys.range(u + 1..).next().map(|&v| from_order_bits(v) - k);
        for (gap, other) in [(below, true), (above, false)] {
            let Some(gap) = gap else { continue };
            if gap < SEP_FLOOR {
                self.under_floor += 1;
            }
            if gap < self.worst {
                self.worst = gap;
                self.worst_at = Some(if other { [k - gap, k] } else { [k, k + gap] });
            }
        }
    }

    /// Summing across heads: the counts add and the worst is the worst, but
    /// the key sets stay apart — two heads' keys are different quantities and
    /// their proximity means nothing.
    pub fn merge(&mut self, other: &SepWitness) {
        self.total += other.total;
        self.repeats += other.repeats;
        self.distinct += other.distinct;
        self.under_floor += other.under_floor;
        if other.worst < self.worst {
            self.worst = other.worst;
            self.worst_at = other.worst_at;
        }
    }

    /// The window the measured separation buys, at the shipped offset spread.
    pub fn window(&self) -> Option<f64> {
        window(self.worst, MARK_SPREAD)
    }

    /// Whether every pair of keys this head holds clears the floor, so that
    /// `ALM.HullSep.lineEval_markKey_lt_of_sep` applies to it.
    pub fn clears_the_floor(&self) -> bool {
        self.under_floor == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_floor_is_the_square_root_of_the_offset_spread() {
        assert_eq!(SEP_FLOOR * SEP_FLOOR, MARK_SPREAD);
        // `ALM.HullSep.the_shipped_separation_floor`, both halves.
        assert!(0.657 * 0.657 < MARK_SPREAD);
        assert!(MARK_SPREAD < 0.659 * 0.659);
    }

    #[test]
    fn the_window_is_the_unit_one_at_unit_separation() {
        // `ALM.HullNear.the_shipped_window`: `(1 - A) / 2 > 0.283`.
        let w = window(1.0, MARK_SPREAD).unwrap();
        assert!((w - (1.0 - MARK_SPREAD) / 2.0).abs() < 1e-15);
        assert!(w > 0.283);
        // And it narrows with the separation, to nothing at the floor.
        assert!(window(0.8, MARK_SPREAD).unwrap() < w);
        assert_eq!(window(SEP_FLOOR, MARK_SPREAD), None);
        assert_eq!(window(0.65, MARK_SPREAD), None);
    }

    #[test]
    fn unit_keys_clear_the_floor_and_the_worst_gap_is_the_unit() {
        let mut w = SepWitness::new();
        for k in [3.0, 1.0, 2.0, -1.0, 0.0] {
            w.observe(k);
        }
        assert_eq!((w.total, w.distinct, w.repeats, w.under_floor), (5, 5, 0, 0));
        assert_eq!(w.worst, 1.0);
        assert!(w.clears_the_floor());
        assert!(w.window().unwrap() > 0.283);
    }

    #[test]
    fn a_rewrite_of_the_same_key_is_not_a_separation_failure() {
        let mut w = SepWitness::new();
        for k in [7.0, 7.0, 7.0, 8.0] {
            w.observe(k);
        }
        assert_eq!((w.distinct, w.repeats, w.under_floor), (2, 2, 0));
        assert_eq!(w.worst, 1.0);
    }

    #[test]
    fn a_pair_under_the_floor_is_caught_wherever_it_arrives() {
        // The offender arrives last, between two keys already held.
        let mut w = SepWitness::new();
        for k in [0.0, 2.0, 0.6] {
            w.observe(k);
        }
        assert_eq!(w.under_floor, 1);
        assert_eq!(w.worst, 0.6);
        assert_eq!(w.worst_at, Some([0.0, 0.6]));
        assert!(!w.clears_the_floor());
        assert_eq!(w.window(), None);

        // And the same pair arrives in the other order, the offender first.
        let mut v = SepWitness::new();
        for k in [0.6, 2.0, 0.0] {
            v.observe(k);
        }
        assert_eq!((v.under_floor, v.worst), (1, 0.6));
        assert_eq!(v.worst_at, Some([0.0, 0.6]));
    }

    #[test]
    fn the_minimum_is_over_every_pair_and_not_only_over_the_adjacent_ones() {
        // Inserting 5 between 0 and 10 makes the old adjacency non-adjacent;
        // the running minimum must still be the smallest gap that ever held.
        let mut w = SepWitness::new();
        for k in [0.0, 10.0, 5.0, 4.9, 9.0] {
            w.observe(k);
        }
        assert!((w.worst - 0.1).abs() < 1e-12);
        assert_eq!(w.under_floor, 1);
    }

    #[test]
    fn a_non_finite_key_is_counted_and_dropped_rather_than_ordered() {
        let mut w = SepWitness::new();
        for k in [f64::NAN, 1.0, f64::INFINITY, 2.0] {
            w.observe(k);
        }
        assert_eq!((w.total, w.distinct, w.worst), (4, 2, 1.0));
    }

    #[test]
    fn merging_two_heads_keeps_the_worst_and_never_mixes_their_keys() {
        let mut a = SepWitness::new();
        for k in [0.0, 1.0] {
            a.observe(k);
        }
        let mut b = SepWitness::new();
        for k in [0.5, 1.5] {
            b.observe(k);
        }
        a.merge(&b);
        // 0.5 sits between b's keys but is half a step from a's, and that
        // proximity is not a fact about either head.
        assert_eq!((a.total, a.distinct, a.worst), (4, 4, 1.0));
        assert_eq!(a.under_floor, 0);
    }

    #[test]
    fn the_order_is_total_and_survives_the_sign() {
        let xs = [-3.0, -0.0, 0.0, 1e-300, -1e300, 1e300, 2.5];
        let mut ordered: Vec<f64> = xs.to_vec();
        ordered.sort_by(|p, q| p.partial_cmp(q).unwrap());
        let mut bits: Vec<u64> = xs.iter().map(|&x| order_bits(x)).collect();
        bits.sort_unstable();
        let back: Vec<f64> = bits.iter().map(|&u| from_order_bits(u)).collect();
        assert!(ordered.iter().zip(&back).all(|(p, q)| p == q));
    }
}

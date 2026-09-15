//! The integer key behind a live point, and the comparison that does not round.
//!
//! `todo3.md` section 4 measures the wall at `94 906 266 = ceil(sqrt(2^53))`
//! and says where it is: in the representation, not in the arithmetic.  The
//! key `k` reaches a head as the point `(2k, -k^2 + d)`, and past that
//! magnitude `-k^2` is not an exact double, so the head is handed the wrong
//! point.  Exact scoring on the stored points -- `exact::dot_cmp`, an 80-bit
//! accumulator, `f128` -- returns the same first failure to the unit, because
//! none of them recover a coordinate that was rounded before it arrived.
//!
//! But `2k` is exact up to `k = 2^52` -- an integer of one more bit than the
//! key, so the format holds it and holds the half of it that recovers the key
//! (`ALM.LiftCompare.the_abscissa_and_its_half_are_both_storable`) -- and the
//! second coordinate is a function of the first: for a live key it is `-k^2`
//! plus a recency term under one.  So the head does not have to read it.
//! Recovering `k` from the abscissa and rebuilding the comparison from `k`
//! moves the wall from `2^26.5` to `2^52`, in the same 2D, at the same
//! `O(log n)`, and without a wider embedding --
//! which is the one thing `todo3.md` section 4 leaves as the alternative, and
//! it is a change to the weights.
//!
//! What licenses reading the ordinate off the abscissa is the family: the
//! offset `d` lives in `[0, MARK_SPREAD]` with `MARK_SPREAD < 1/2`
//! (`ALM.LiftCompare.the_shipped_spread`, the shipped constant; bounded by one
//! in `ALM.HullMark.marked_sep_of_shipped`), so two live offsets differ by less
//! than one and cannot bridge the unit gap between two integer keys.  That is
//! the same inequality `ALM.HullMark.not_eraseStep_of_marked` spends to show
//! the container drops no key, and `ALM.MarkedPosition.markKey_not_concurrent`
//! spends to put the marked keys in general position.  A point whose offset is
//! outside that range is not a key this file can speak for, and `of` returns
//! `None` for it -- which is also how the cleared family (`d ~ -1e30`) and the
//! flat one (`ky = 1`) are turned away, with no separate test.
//!
//! The query is put at unit scale by `on_the_grid`
//! (`ALM.QueryScale.onTheGrid_scaleQuery` returns the lifted `(q, 1)` on the
//! nose), but the division rounds, so what arrives is an integer plus a
//! residual of an ulp or two rather than an integer.  The residual is kept,
//! not rounded away: `ALM.LiftResidual.upper_near_lt_iff` is the same
//! equivalence at `z0 + eps`, and it holds for every `|eps| < 1/2 - A`, which
//! is `RESIDUAL_WINDOW`.  The keys being integers is what pays for it -- the
//! score moves by `2 eps (k - k')` but the order does not, because that same
//! step divides out against the integer gap between the squared distances.  A
//! query outside the window is refused and the caller falls back to
//! `exact::dot_cmp`, which is exact in the stored points and therefore no
//! better and no worse than the hull head.

use core::cmp::Ordering;

use crate::exact;
use crate::lift::MARK_SPREAD;

/// The largest `|k|` recoverable from the abscissa `2k`.
///
/// Doubles are one apart below `2^53`, so `2k` is exact and `2k / 2` returns
/// `k` for every `|k| <= 2^52`.  Above it the abscissa itself is rounded and
/// there is nothing left to recover.  Both halves of that are
/// `ALM.LiftCompare.the_abscissa_and_its_half_are_both_storable`, and
/// `ALM.LiftCompare.the_section_4b_key` is one shipped key at which the
/// abscissa is stored and the ordinate is not.
pub const KEY_LIMIT: i64 = 1 << 52;

/// A live key, as the integer it is rather than as the point it arrived in.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct LiftKey {
    /// `kx / 2`, exactly.
    pub v: i64,
    /// `ky + k^2`: the recency term the compiler added, or zero where the
    /// wall swallowed it (`ALM.HullWall.markKey_eq_liftKey_of_wall`).
    pub delta: f64,
}

impl LiftKey {
    /// Recover the integer key, or refuse the point.
    ///
    /// Three things are checked and each rejects a family: a non-integral or
    /// out-of-range abscissa, which no `embed_key` branch produces and no
    /// theorem covers; and an offset outside `[-MARK_SPREAD, MARK_SPREAD]`,
    /// which is the cleared key, the flat key, and any point off the
    /// paraboloid altogether.
    ///
    /// The offset is recovered exactly.  `ky` is within one of `-k^2`, so `ky`
    /// and `k * k` are within a factor of two of each other and their sum is
    /// exact by Sterbenz's lemma -- which also means `point` below rebuilds
    /// the double the compiler emitted, bit for bit.
    pub fn of(k: [f64; 2]) -> Option<LiftKey> {
        let v = k[0] / 2.0;
        if v != v.round() || v.abs() > KEY_LIMIT as f64 {
            return None;
        }
        let delta = k[1] + v * v;
        // A positive test, so that a `NaN` ordinate fails it rather than passing
        // a negated one.
        (delta.abs() <= MARK_SPREAD).then_some(LiftKey { v: v as i64, delta })
    }

    /// The point the compiler emitted, rebuilt from the integer.
    pub fn point(self) -> [f64; 2] {
        let v = self.v as f64;
        [2.0 * v, -(v * v) + self.delta]
    }
}

/// How far off the integer the query's abscissa may be and still be answered
/// on the integers: `1/2 - MARK_SPREAD`, about `0.0672`.
///
/// Two offsets spread by `MARK_SPREAD` and a residual of `eps` between them
/// have to leave a unit of integer separation standing, which is
/// `2 A + 2 |eps| < 1`.  `ALM.LiftResidual.the_shipped_window` is this number
/// at the shipped spread, and the released programs leave a residual of one
/// ulp -- `1.9e-6` at a query of `2^33`, five orders inside it.
pub const RESIDUAL_WINDOW: f64 = 0.5 - MARK_SPREAD;

/// A query at unit scale, as the integer it was meant to be and the residual
/// the normalisation left on it.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct UnitQuery {
    /// The nearest integer to the abscissa.
    pub qx: i64,
    /// `qx_f64 - qx`, exactly: the two are within half a unit of each other,
    /// so the subtraction is exact by Sterbenz's lemma and the residual is the
    /// whole of what the division rounded.
    pub eps: f64,
    /// `qy == 1`, the upper envelope.  `false` is `qy == -1`, where the score
    /// is convex in the key and the maximum sits at an end.
    pub upper: bool,
}

impl UnitQuery {
    /// Read a normalised query, or refuse it.
    ///
    /// `on_the_grid` divides the compiler's scale out, and the division is a
    /// rounding: at a query of `2^33` the quotient misses the integer by an
    /// ulp.  Refusing those was the whole of `todo3.md` section 4a's residual
    /// cost -- 64 queries of the `addition` run fell back to the stored
    /// points, on the one head whose ordinate the wall had already eaten.
    /// They are not refused now: the residual is carried into the comparison,
    /// where `ALM.LiftResidual.upper_near_lt_iff` says it changes the order
    /// only between two keys the query is equidistant from.
    pub fn of(q: [f64; 2]) -> Option<UnitQuery> {
        let upper = if q[1] == 1.0 {
            true
        } else if q[1] == -1.0 {
            false
        } else {
            return None;
        };
        let qx = q[0].round();
        if qx.abs() > KEY_LIMIT as f64 {
            return None;
        }
        // A positive test, so a `NaN` abscissa is refused here rather than
        // rounded into an integer above.
        let eps = q[0] - qx;
        (eps.abs() < RESIDUAL_WINDOW).then_some(UnitQuery { qx: qx as i64, eps, upper })
    }

    /// Which of two live keys this query scores higher.
    ///
    /// At `qy = 1` the score is `2 q k - k^2 + d = -(k - q)^2 + q^2 + d`, so
    /// the order is the order of `(k - q)^2` reversed, and the offsets decide
    /// only where those are equal -- they cannot do more, because two of them
    /// differ by less than one and `(k - q)^2` is an integer.  That is
    /// `ALM.LiftCompare.upper_lt_iff`, and `lower_lt_iff` is the `qy = -1`
    /// branch, where the sign flips and the same expression is `(k + q)^2`
    /// maximised instead with the offsets reversed.  Both are stated as
    /// equivalences, so the two arms below are the comparison and not a
    /// sufficient condition for it; `ALM.HullNear` proves the same win from a
    /// query near a key, which is the weaker thing a container cannot use.
    ///
    /// `(k - q)^2` is computed in `i128` and never rounds: at the limit above
    /// it is `2^106` (`ALM.LiftCompare.sq_dist_le`), twenty bits inside the
    /// width (`sq_dist_fits`).  This is the whole of the move from `2^26.5`
    /// to `2^52`.
    ///
    /// The residual enters only where those integers tie, and there it is the
    /// comparison: `ALM.LiftResidual.upper_lt_iff_of_sq_eq` says two keys the
    /// query is equidistant from are separated by `2 eps (k - k')` and the two
    /// offsets, with no bound on `eps` needed, since that is an identity and
    /// not an estimate.  Its four terms are summed exactly by
    /// `exact::expansion_sign`: `2 eps` is exact and `k - k'` is at most
    /// `2^53`, so the product is a `two_prod` and nothing rounds here either.
    pub fn cmp(self, a: LiftKey, b: LiftKey) -> Ordering {
        let sq = |v: i64| {
            let t = (if self.upper { v - self.qx } else { v + self.qx }) as i128;
            t * t
        };
        let (sa, sb) = (sq(a.v), sq(b.v));
        if sa != sb {
            return if self.upper { sb.cmp(&sa) } else { sa.cmp(&sb) };
        }
        let (p, e) = exact::two_prod(2.0 * self.eps, (a.v - b.v) as f64);
        let (da, db) = if self.upper { (a.delta, b.delta) } else { (b.delta, a.delta) };
        exact::expansion_sign(&[p, e, da, -db])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::dot_cmp;

    /// `embed_key`'s recency term, as the compiler computes it.
    fn offset(pos: f64) -> f64 {
        0.3 * (1.0 / std::f64::consts::LN_2 - 1.0 / (pos + 2.0).ln())
    }

    fn key(k: f64, pos: f64) -> [f64; 2] {
        [2.0 * k, -(k * k) + offset(pos)]
    }

    #[test]
    fn the_integer_and_its_offset_come_back_and_the_point_does_too() {
        for k in [0.0, 1.0, -7.0, 94906266.0, 336860161.0, 4503599627370496.0] {
            for p in [0.0, 40.0, 1e9] {
                let point = key(k, p);
                let lk = LiftKey::of(point).expect("a live key at {k}");
                assert_eq!(lk.v as f64, k, "the integer");
                assert_eq!(lk.point(), point, "and the double it arrived as");
            }
        }
    }

    #[test]
    fn the_other_families_are_refused_rather_than_rounded_into_this_one() {
        assert_eq!(LiftKey::of([0.0, 1.0]), None, "the flat key of `fetch_sum`");
        assert_eq!(LiftKey::of([14.0, 1.0]), None, "and a varying one");
        assert_eq!(LiftKey::of([14.0, -49.0 - 1e30]), None, "the clear marker");
        assert_eq!(LiftKey::of([15.0, -56.25]), None, "a half-integer key");
        assert_eq!(LiftKey::of([14.0, -49.0 + 5.0]), None, "an offset no branch adds");
        assert_eq!(LiftKey::of([f64::INFINITY, 0.0]), None);
        assert_eq!(LiftKey::of([2.0, f64::NAN]), None);
        // Past `2^52` the abscissa is rounded and there is nothing to recover.
        let past = (KEY_LIMIT as f64) * 2.0;
        assert_eq!(LiftKey::of([2.0 * past, -(past * past)]), None);
    }

    #[test]
    fn at_the_wall_the_stored_points_tie_and_the_integers_do_not() {
        // `todo3.md` section 4: the first query whose own key stops being the
        // strict argmax, under rounded scoring and under exact scoring on the
        // stored points alike.  Here the same query, the same two keys, and
        // the comparison rebuilt from the integer.
        let q = 94906266i64;
        let (a, b) = (key(q as f64, 0.0), key(q as f64 + 1.0, 0.0));
        let qf = [q as f64, 1.0];
        assert_eq!(dot_cmp(qf, a, b), Ordering::Equal, "exact in the stored points: a tie");

        let uq = UnitQuery::of(qf).expect("an integer query at unit scale");
        let (la, lb) = (LiftKey::of(a).unwrap(), LiftKey::of(b).unwrap());
        assert_eq!(uq.cmp(la, lb), Ordering::Greater, "and the key is its own argmax");
    }

    #[test]
    fn past_the_wall_the_shipped_key_of_section_4b_is_separated_again() {
        // `0x20202020 = 673720322` is the abscissa `todo3.md` section 4b names,
        // so the key is half of it and its square is 26 binades past `2^53`:
        // the stored ordinate is quantised to 32 and a unit gap is invisible.
        let v = 336860161i64;
        let (a, b) = (key(v as f64, 1e6), key(v as f64 + 1.0, 1e6));
        assert_eq!(a[1], -((v * v) as f64), "the offset did not survive the store");
        let qf = [v as f64, 1.0];
        // And the failure here is not a tie but an inversion: the quantised
        // ordinates put the neighbour ahead of the query's own key, so no
        // tie-break can recover it -- the head answers with a key that does
        // not win.  This is `ScoreGaps::observe_misranked` on the shipped
        // model, and the reason section 4b is a defect and not a caveat.
        // `ALM.LiftCompare.the_section_4b_query` is the same pair decided the
        // other way, from the abscissa, at any offsets the compiler emits.
        assert_eq!(dot_cmp(qf, a, b), Ordering::Less, "the stored points are misranked");

        let uq = UnitQuery::of(qf).unwrap();
        assert_eq!(uq.cmp(LiftKey::of(a).unwrap(), LiftKey::of(b).unwrap()), Ordering::Greater);
    }

    #[test]
    fn the_recency_term_decides_only_a_symmetric_pair() {
        // Two keys straddling the query at equal distance: the integers tie and
        // the later write wins, which is the whole of what the offset is for.
        let (early, late) = (key(9.0, 10.0), key(11.0, 1e9));
        let uq = UnitQuery::of([10.0, 1.0]).unwrap();
        let (a, b) = (LiftKey::of(early).unwrap(), LiftKey::of(late).unwrap());
        assert!(a.delta < b.delta, "the later position carries the larger offset");
        assert_eq!(uq.cmp(a, b), Ordering::Less);
        // And one step off centre the distance decides instead, against it.
        let uq = UnitQuery::of([9.0, 1.0]).unwrap();
        assert_eq!(uq.cmp(a, b), Ordering::Greater);
    }

    #[test]
    fn the_lower_envelope_maximises_the_far_key_and_reverses_the_offset() {
        // At `qy = -1` the score is `(k + q)^2 - q^2 - d`, convex in the key.
        let uq = UnitQuery::of([10.0, -1.0]).unwrap();
        let (near, far) = (LiftKey::of(key(1.0, 0.0)).unwrap(), LiftKey::of(key(5.0, 0.0)).unwrap());
        assert_eq!(uq.cmp(near, far), Ordering::Less, "the far key wins");
        // A symmetric pair about `-q`, where the smaller offset now wins.
        let (early, late) = (LiftKey::of(key(-9.0, 10.0)).unwrap(), LiftKey::of(key(-11.0, 1e9)).unwrap());
        assert_eq!(uq.cmp(early, late), Ordering::Greater);
    }

    #[test]
    fn a_query_the_grid_has_not_normalised_is_refused() {
        assert_eq!(UnitQuery::of([10.0, 14142135623.730951]), None, "the compiler's scale");
        assert_eq!(UnitQuery::of([10.5, 1.0]), None, "an abscissa between two keys");
        assert_eq!(UnitQuery::of([10.0, 0.0]), None, "and the degenerate direction");
        assert_eq!(UnitQuery::of([10.0, 1.0]), Some(UnitQuery { qx: 10, eps: 0.0, upper: true }));
        assert_eq!(UnitQuery::of([10.0, -1.0]), Some(UnitQuery { qx: 10, eps: 0.0, upper: false }));
        assert_eq!(UnitQuery::of([f64::NAN, 1.0]), None);
    }

    #[test]
    fn the_division_leaves_a_residual_and_the_residual_is_carried() {
        // `todo3.md` section 4a: the query the `addition` run asks 64 times,
        // where the scale divided back out lands an ulp short of the integer.
        // The old `UnitQuery::of` refused it and the head fell back to the
        // stored points -- on the one head past the wall, where those points
        // have already lost the unit that separates two keys.
        let q = [4294967291.0000005, 1.0];
        let u = UnitQuery::of(q).expect("an ulp off the grid is still on it");
        assert_eq!(u.qx, 4294967291);
        assert_eq!(u.eps, q[0] - 4294967291.0, "and the residual is exact");
        assert!(u.eps.abs() < RESIDUAL_WINDOW && u.eps != 0.0, "{} ", u.eps);
        // Half a step is not a residual but a different query, and the window
        // stops well short of it.
        assert_eq!(UnitQuery::of([10.07, 1.0]), None, "outside `1/2 - MARK_SPREAD`");
        const { assert!(RESIDUAL_WINDOW > 0.067) }; // `ALM.LiftResidual.the_shipped_window`
    }

    #[test]
    fn inside_the_window_the_residual_moves_the_score_and_not_the_order() {
        // The score at these keys moves by `2 eps (k - k')`, which here is
        // thousands of units; the answer does not move at all.
        let (a, b) = (key(336860161.0, 1e6), key(336860162.0, 1e6));
        let (la, lb) = (LiftKey::of(a).unwrap(), LiftKey::of(b).unwrap());
        for eps in [0.0, 1e-6, -1e-6, 0.06, -0.06] {
            let u = UnitQuery::of([336860161.0 + eps, 1.0]).expect("inside the window");
            assert_eq!(u.cmp(la, lb), Ordering::Greater, "at eps = {eps}");
        }
    }

    #[test]
    fn the_symmetric_pair_is_where_the_residual_decides() {
        // Two keys equidistant from the integer: there the offsets decide, and
        // the residual is weighed against them, multiplied by the step between
        // the keys -- so a residual of `1e-4` over a step of two is nothing
        // beside an offset gap of `0.11`, and one of `0.05` overturns it.
        // `ALM.LiftResidual.upper_lt_iff_of_sq_eq` is that comparison exactly.
        let (early, late) = (key(9.0, 10.0), key(11.0, 1e9));
        let (a, b) = (LiftKey::of(early).unwrap(), LiftKey::of(late).unwrap());
        assert!(b.delta - a.delta > 0.1, "the later position carries the larger offset");
        assert_eq!(UnitQuery::of([10.0, 1.0]).unwrap().cmp(a, b), Ordering::Less, "the offset");
        assert_eq!(UnitQuery::of([10.0001, 1.0]).unwrap().cmp(a, b), Ordering::Less, "and with it");
        assert_eq!(UnitQuery::of([9.95, 1.0]).unwrap().cmp(a, b), Ordering::Greater, "against it");
        // And the exact dot product on the stored points agrees, these keys
        // being far below the wall.
        for qx in [10.0, 10.0001, 9.95, 10.05] {
            let u = UnitQuery::of([qx, 1.0]).unwrap();
            assert_eq!(u.cmp(a, b), dot_cmp([qx, 1.0], early, late), "at {qx}");
        }
    }

    #[test]
    fn below_the_wall_it_answers_what_the_stored_points_answer() {
        // Where the old comparison is sound the new one agrees with it, over
        // every pair of a spread of keys and queries.  Nothing else would be
        // worth having: the point is to move the wall, not the semantics.
        let ks: Vec<i64> = (-20..20).chain([1000, 94906265, -94906265]).collect();
        let mut pairs = 0;
        for (i, &a) in ks.iter().enumerate() {
            for &b in &ks[i + 1..] {
                let (pa, pb) = (key(a as f64, 7.0), key(b as f64, 9.0));
                let (la, lb) = (LiftKey::of(pa).unwrap(), LiftKey::of(pb).unwrap());
                for qx in [-30i64, -1, 0, 1, 17, 1000] {
                    for eps in [0.0, 1e-9, -1e-9, 0.05, -0.05] {
                        for qy in [1.0, -1.0] {
                            let q = [qx as f64 + eps, qy];
                            let uq = UnitQuery::of(q).unwrap();
                            assert_eq!(uq.cmp(la, lb), dot_cmp(q, pa, pb), "{a} vs {b} at {q:?}");
                            pairs += 1;
                        }
                    }
                }
            }
        }
        assert!(pairs > 50_000, "only {pairs} comparisons");
    }
}

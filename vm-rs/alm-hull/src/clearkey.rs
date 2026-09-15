//! The cleared entry: recognised, bounded, and then not read again.
//!
//! `embed_key` subtracts `BIG = 1e30` from the ordinate when the clear flag is
//! set (`transformer_vm/graph/core.py:314`), and the point goes into the same
//! container as every live key.  `lifthead.rs` cannot hold it as a key: the
//! integer path reads the ordinate off the abscissa, and for a cleared entry
//! that relation is gone.  `ALM.ClearKey.the_marker_costs_the_grid` says how
//! completely: past `10^30` the representables are `2^47` apart, so the recency
//! term, the low bits of `-k^2`, and the unit step the whole tie argument runs
//! on are all below one ulp of the stored ordinate.
//!
//! What `ALM.ClearKey` offers instead is that the entry never has to be read.
//! `marked_sup'_eq_live` says the maximum over live and cleared entries
//! together *is* the maximum over the live ones alone, under three hypotheses:
//! a live entry exists, every base score is inside `+/-M`, and `2M < B`.  None
//! of the three is free at `2^52`, where a score reaches `2^106` and the third
//! plainly fails, so `ClearGuard` carries the bounds and tests them at each
//! query rather than assuming them.
//!
//! The query enters the hypotheses too.  `dot_marked` is stated at `qy = 1`,
//! where the marker passes into the score undiminished; at a general `qy` it
//! arrives as `qy * B`, so the test is `2M < qy * B`, and a query with
//! `qy <= 0` has no margin at all -- at `qy < 0` the marker is *added* to the
//! score and the cleared entry wins every time, which is a fact about the
//! released representation and not something a container may round away.

use crate::grid::ulp;
use crate::lift::CLEAR_MARK;

/// A cleared entry, as much of it as survives the marker.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ClearKey {
    /// A bound on `|ky|` as it stood before `BIG` was taken off it.
    ///
    /// A bound and not the value: the subtraction rounded, and adding `BIG`
    /// back rounds again, so `ky + BIG` is the original to within an ulp of
    /// the stored ordinate and no closer.  A bound is the whole of what
    /// `marked_sup'_eq_live` asks for -- it wants `M`, not the score -- so a
    /// bound is what this is, padded by that ulp.
    pub base_bound: f64,
}

impl ClearKey {
    /// Recognise the marker, or refuse the point.
    ///
    /// The test is `lift.rs`'s: an ordinate below `-BIG/2`.  It is only sound
    /// *after* `LiftKey::of` has refused the point, and the reason is the new
    /// range rather than the old one -- a live key with `|k| > 7.1e14` has
    /// `-k^2` below `-BIG/2` and is nothing to do with clearing, so the live
    /// family has to be taken out first and `lifthead.rs` takes it out first.
    ///
    /// Which family a refused point *really* belongs to is not something this
    /// has to get right, and that is worth saying plainly: the argument
    /// downstream is `2M < qy * B` over a bound `M` this point is inside, and
    /// any point whatsoever with an ordinate that low is dominated by it.  The
    /// classification decides where a point is stored, not whether the answer
    /// is correct.
    pub fn of(k: [f64; 2]) -> Option<ClearKey> {
        if !k[0].is_finite() || !k[1].is_finite() || k[1] >= -CLEAR_MARK / 2.0 {
            return None;
        }
        Some(ClearKey { base_bound: (k[1] + CLEAR_MARK).abs() + ulp(k[1]) })
    }
}

/// The bounds `ALM.ClearKey.marked_sup'_eq_live` is stated over, accumulated
/// as the keys arrive.
///
/// One number per coordinate rather than a bound on the score itself: the
/// score depends on the query, and the query is not there yet at insert time.
/// `bound` puts the two together in a multiply-add when it is.
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct ClearGuard {
    /// `max |kx|` over every entry the head holds, live and cleared alike.
    max_kx: f64,
    /// `max |ky|` over the same, the cleared ones with the marker taken off.
    max_ky: f64,
}

impl ClearGuard {
    /// Take one entry into the bounds: its abscissa, and its ordinate as it
    /// would be with no marker on it.
    pub fn observe(&mut self, kx: f64, base_ky: f64) {
        if kx.abs() > self.max_kx {
            self.max_kx = kx.abs();
        }
        if base_ky.abs() > self.max_ky {
            self.max_ky = base_ky.abs();
        }
    }

    /// `M` of the theorem at this query: a bound on the base score of every
    /// entry the head holds, cleared entries included.
    pub fn bound(&self, q: [f64; 2]) -> f64 {
        q[0].abs() * self.max_kx + q[1].abs() * self.max_ky
    }

    /// Whether this query's answer can leave the cleared entries out.
    ///
    /// `hB` of `marked_sup'_eq_live`, at the marker this query actually sees.
    /// Written as a positive test, so an infinite bound or a `NaN` query fails
    /// it and the cleared entries are consulted -- which is the safe side.
    /// The remaining hypothesis, `hL`, is the caller's: this says nothing at
    /// all about a head with no live entry in it.
    pub fn dominated(&self, q: [f64; 2]) -> bool {
        q[1] > 0.0 && 2.0 * self.bound(q) < q[1] * CLEAR_MARK
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::liftkey::LiftKey;

    /// A cleared key as `embed_key` writes it.
    fn cleared(k: f64, pos: f64) -> [f64; 2] {
        let off = 0.3 * (1.0 / std::f64::consts::LN_2 - 1.0 / (pos + 2.0).ln());
        [2.0 * k, -(k * k) + off - CLEAR_MARK]
    }

    #[test]
    fn the_marker_is_recognised_and_the_ordinate_under_it_is_bounded() {
        for k in [0.0, 1.0, -7.0, 336860161.0] {
            let p = cleared(k, 40.0);
            assert_eq!(LiftKey::of(p), None, "the live family refuses it first");
            let c = ClearKey::of(p).expect("and this one takes it");
            assert!(c.base_bound >= k * k, "{} bounds |ky| = {}", c.base_bound, k * k);
            // The bound is an ulp of `1e30` wide and no wider: the marker
            // costs the low bits and nothing beyond them.
            assert!(c.base_bound <= k * k + 1.0 + 2.0 * ulp(CLEAR_MARK));
        }
    }

    #[test]
    fn a_live_key_is_not_a_cleared_one_however_large_it_is() {
        // `-k^2` past `-BIG/2` on its own: the point this test exists for.
        let k = 1e15;
        let p = [2.0 * k, -(k * k)];
        assert!(p[1] < -CLEAR_MARK / 2.0, "it is below the marker's threshold");
        assert!(LiftKey::of(p).is_some(), "and still a live key, taken first");
    }

    #[test]
    fn the_margin_holds_at_the_shipped_keys_and_fails_at_the_top_of_the_range() {
        let mut g = ClearGuard::default();
        // Section 4b: keys around `3.4e8`, queries around `8.6e9`.
        g.observe(673720322.0, -1.1347476806894592e17);
        assert!(g.dominated([8589934503.0, 1.0]), "thirteen orders of margin");
        // And the top of the new range, where `2^106` is past `BIG`.
        let v = (1i64 << 52) as f64;
        g.observe(2.0 * v, -(v * v));
        assert!(!g.dominated([1.0, 1.0]), "the score is past the marker itself");
    }

    #[test]
    fn a_query_that_adds_the_marker_rather_than_subtracting_it_is_never_dominated() {
        let mut g = ClearGuard::default();
        g.observe(4.0, -4.0);
        assert!(g.dominated([1.0, 1.0]));
        assert!(!g.dominated([1.0, -1.0]), "at qy < 0 the cleared entry wins");
        assert!(!g.dominated([1.0, 0.0]), "and at qy = 0 it ties");
        assert!(!g.dominated([f64::NAN, 1.0]));
    }

    #[test]
    fn the_other_families_are_not_taken_for_a_marker() {
        assert_eq!(ClearKey::of([0.0, 1.0]), None, "the flat key");
        assert_eq!(ClearKey::of([14.0, -49.0]), None, "a small live key");
        assert_eq!(ClearKey::of([14.0, f64::NEG_INFINITY]), None);
        assert_eq!(ClearKey::of([f64::NAN, -1e30]), None);
    }
}

//! Which of the three key families each arriving key belongs to.
//!
//! `todo3.md` section 4's group: every theorem about the container's size
//! assumes the keys are lifts, `k -> (2k, -k^2)`, and the compiler never emits
//! one.  `alm-compile/src/graph.rs`'s `embed_key` builds `kx = 2k` and then
//! puts one of three things in `ky`:
//!
//! ```text
//! TieBreak::Latest    ky = -k^2 - clear * BIG + LATEST_ALPHA * inv_log_pos(p)
//! TieBreak::Average   ky = 1                  -- the lift discarded, kx kept
//! ```
//!
//! So the three families are the live key, the cleared key and the flat key,
//! and which one a head is holding decides which theorem applies to it.
//! `ALM.HullMark.Marked` is the live family — `not_eraseStep_of_marked` says
//! the recency term is too small to make the container drop a key, so the
//! container is the set of keys inserted, `build_card_eq_of_marked`.
//! `ALM.HullClear.not_marked_of_clearKey` is the cleared one, and it refutes
//! `Marked`: `eraseStep_of_clearKey` shows a cleared key bracketed by live ones
//! *is* erased, which is how it leaves.  The flat family is what `fetch_sum`
//! builds — with a constant `k` it is `ALM.CumSum`, every entry tied and the
//! head returning the mean — and no hull statement is about it, because with
//! `ky` constant the envelope is a line and not a hull at all.
//!
//! The offset's own range is `[0, LATEST_ALPHA / log 2)`, because
//! `inv_log_pos(p) = 1/log 2 - 1/log(p + 2)` rises to `1/log 2` and no further;
//! `ALM.HullMark.marked_sep_of_shipped` is that number under one, which is the
//! separation `not_eraseStep_of_marked` needs on unit-separated keys.
//!
//! And `ALM.HullWall.markKey_eq_liftKey_of_wall` is why the classification is
//! not a formality: past `2^53` the representable doubles are two apart and the
//! offset is under one, so a key past the wall carries *no* offset — the live
//! family degenerates to the pure lift (`lifted_of_marked_of_wall`) and the
//! separation the compiler paid for is gone (`marks_tie_past_the_wall`).  That
//! is the difference between `Lifted` and `Marked` below, and it is counted.

use crate::grid::{ulp, GRID_LIMIT};

/// `LATEST_ALPHA / log 2`: the supremum of the recency offset `embed_key` adds
/// to a live key, over every position the model can reach.
pub const MARK_SPREAD: f64 = 0.3 / std::f64::consts::LN_2;

/// `BIG` of `embed_key`, subtracted from `ky` when the clear flag is set.
pub const CLEAR_MARK: f64 = 1e30;

/// Which family one key belongs to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Family {
    /// `(2k, -k^2)` on the nose: either position zero, where `inv_log_pos` is
    /// zero, or a key past the wall, where the offset cannot be stored.
    Lifted,
    /// `(2k, -k^2 + d)` with `0 < d <= MARK_SPREAD`: the live key of
    /// `ALM.HullMark.Marked`, recency term and all.
    Marked,
    /// `ky` below `-BIG/2`: the clear marker of `ALM.HullClear.clearKey`.
    Cleared,
    /// `ky == 1`: `TieBreak::Average` threw the lift away and kept `kx`.
    Flat,
    /// None of the above — a key no theorem here covers.
    Off,
}

/// What a run's keys looked like.
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct LiftWitness {
    /// Keys examined: one per head per layer step.
    pub total: usize,
    /// How many were in each family.
    pub lifted: usize,
    pub marked: usize,
    pub cleared: usize,
    pub flat: usize,
    pub off: usize,
    /// Cross-cutting, over every non-flat key: how many had a non-integer
    /// `kx / 2`.  `ALM.HullMark.Marked.perturbed` asks for `z : Z`, and
    /// `not_eraseStep_of_marked` spends it — the keys have to be a unit apart
    /// for an offset under one to be unable to bridge them.
    pub noninteger: usize,
    /// And how many had `k^2` past `2^53`, where
    /// `ALM.HullWall.markKey_eq_liftKey_of_wall` forces the offset to zero.
    pub past_wall: usize,
    /// The largest live offset seen, and the key that carried it.  A value
    /// above `MARK_SPREAD` would be a key the compiler did not build.
    pub worst: f64,
    pub worst_at: Option<[f64; 2]>,
}

impl LiftWitness {
    pub fn observe(&mut self, k: [f64; 2]) -> Family {
        self.total += 1;
        // `ky = 1` is the `Average` branch of `embed_key` verbatim, and no
        // other branch can reach it: a live intercept is `-k^2 + d` with
        // `d < MARK_SPREAD < 1`, so it is at most `MARK_SPREAD`, and a cleared
        // one is thirty orders below that.
        if k[1] == 1.0 {
            self.flat += 1;
            return Family::Flat;
        }
        let v = k[0] / 2.0;
        let sq = v * v;
        if !sq.is_finite() {
            self.off += 1;
            return Family::Off;
        }
        if v != v.round() {
            self.noninteger += 1;
        }
        if sq > GRID_LIMIT {
            self.past_wall += 1;
        }
        let delta = k[1] + sq;
        if delta < -CLEAR_MARK / 2.0 {
            self.cleared += 1;
            return Family::Cleared;
        }
        // `delta` is formed by an addition that cancels `sq` against itself, so
        // it is only meaningful down to one ulp of `sq`; below that the key is
        // the lift as far as float64 can tell, which is what the wall theorem
        // says it is.
        let tol = ulp(sq);
        if delta.abs() <= tol {
            self.lifted += 1;
            return Family::Lifted;
        }
        if delta > 0.0 && delta <= MARK_SPREAD + tol {
            self.marked += 1;
            if delta > self.worst {
                self.worst = delta;
                self.worst_at = Some(k);
            }
            return Family::Marked;
        }
        self.off += 1;
        Family::Off
    }

    pub fn merge(&mut self, other: &LiftWitness) {
        self.total += other.total;
        self.lifted += other.lifted;
        self.marked += other.marked;
        self.cleared += other.cleared;
        self.flat += other.flat;
        self.off += other.off;
        self.noninteger += other.noninteger;
        self.past_wall += other.past_wall;
        if other.worst > self.worst {
            self.worst = other.worst;
            self.worst_at = other.worst_at;
        }
    }

    /// Whether every key seen was one the Lean argument covers.
    pub fn is_covered(&self) -> bool {
        self.off == 0
    }

    /// Which family a whole head ran in: the one holding a strict majority of
    /// its keys.  Not all of them, because no head is pure.  A clearing head
    /// is live most of the time and cleared when the flag is set, and *every*
    /// head answers its first layer step on `k = (0, 0)` — the residual has
    /// not been written yet, and the zero vector is the lift of the key `0`,
    /// which is what the head duly inserts.  `None` is a head with no majority
    /// at all, and there is no reason for the compiler to build one.
    pub fn regime(&self) -> Option<Family> {
        let half = self.total / 2;
        if self.lifted + self.marked > half {
            Some(Family::Lifted)
        } else if self.flat > half {
            Some(Family::Flat)
        } else if self.cleared > half {
            Some(Family::Cleared)
        } else if self.off > half {
            Some(Family::Off)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `embed_key`'s recency term, as the compiler computes it.
    fn offset(pos: f64) -> f64 {
        0.3 * (1.0 / std::f64::consts::LN_2 - 1.0 / (pos + 2.0).ln())
    }

    fn key(k: f64, pos: f64, clear: f64) -> [f64; 2] {
        [2.0 * k, -(k * k) - clear * CLEAR_MARK + offset(pos)]
    }

    #[test]
    fn the_three_families_are_told_apart() {
        let mut w = LiftWitness::default();
        assert_eq!(w.observe(key(7.0, 0.0, 0.0)), Family::Lifted, "inv_log_pos(0) is zero");
        assert_eq!(w.observe(key(7.0, 40.0, 0.0)), Family::Marked);
        assert_eq!(w.observe(key(7.0, 40.0, 1.0)), Family::Cleared);
        assert_eq!(w.observe([0.0, 1.0]), Family::Flat, "the constant key of `fetch_sum`");
        assert_eq!(w.observe([4.0, 1.0]), Family::Flat, "and a varying one, lift discarded");
        assert_eq!((w.lifted, w.marked, w.cleared, w.flat, w.off), (1, 1, 1, 2, 0));
        assert!(w.is_covered());
    }

    #[test]
    fn the_offset_stays_inside_the_range_the_no_erase_proof_needs() {
        // `ALM.HullMark.marked_sep_of_shipped`: the whole range is under one,
        // which is the gap between two integer keys.  Over every position a
        // trace could reach, no key leaves it.
        let mut w = LiftWitness::default();
        for p in [0.0, 1.0, 1e3, 1e6, 1e12, 1e300] {
            assert_ne!(w.observe(key(1234.0, p, 0.0)), Family::Off, "at position {p}");
        }
        assert!(w.worst < MARK_SPREAD, "{} is the supremum", MARK_SPREAD);
        assert!(w.worst < 1.0, "and the unit separation survives it");
        assert_eq!(w.off, 0);
    }

    #[test]
    fn past_the_wall_the_offset_is_not_in_the_key_at_all() {
        // `ALM.HullWall.markKey_eq_liftKey_of_wall`, measured: `0x20202020` is
        // one of the 32-bit values `todo3.md` section 4b says the WASM heads
        // are keyed on, and at that magnitude the doubles are 64 apart.  The
        // compiler computes an offset of 0.43 and the key that reaches the head
        // is the pure lift.
        let k = 538976288.0f64;
        let mut w = LiftWitness::default();
        assert_eq!(w.observe(key(k, 1e6, 0.0)), Family::Lifted);
        assert_eq!(w.past_wall, 1);
        assert_eq!(-(k * k) + offset(1e6), -(k * k), "the addition does nothing");

        // And two writes to that key, which the compiler separated, arrive as
        // one point: `ALM.HullWall.marks_tie_past_the_wall`.
        assert_eq!(key(k, 1e6, 0.0), key(k, 1e9, 0.0));
        // Below the wall the same two writes are two distinct keys.
        assert_ne!(key(7.0, 1e6, 0.0), key(7.0, 1e9, 0.0));
    }

    #[test]
    fn a_non_integer_key_is_counted_even_when_it_is_a_lift() {
        // The family is right and the hypothesis `z : Z` is not: the count is
        // cross-cutting because `not_eraseStep_of_marked` needs both.
        let mut w = LiftWitness::default();
        assert_eq!(w.observe(key(7.5, 40.0, 0.0)), Family::Marked);
        assert_eq!((w.noninteger, w.marked), (1, 1));
    }

    #[test]
    fn a_key_off_every_family_is_reported_and_not_rounded_into_one() {
        let mut w = LiftWitness::default();
        // An offset far above anything `embed_key` could add.
        assert_eq!(w.observe([14.0, -49.0 + 5.0]), Family::Off);
        // And one below the lift, which no branch of `embed_key` produces.
        assert_eq!(w.observe([14.0, -49.0 - 5.0]), Family::Off);
        assert_eq!(w.off, 2);
        assert!(!w.is_covered());
    }

    #[test]
    fn a_heads_regime_survives_the_first_step_and_the_clear_flag() {
        // Both mixtures the released model actually shows: a flat head whose
        // first key is the unwritten `(0, 0)`, and a live head that clears now
        // and then.
        let mut flat = LiftWitness::default();
        flat.observe([0.0, 0.0]);
        for k in 1..8 {
            flat.observe([2.0 * k as f64, 1.0]);
        }
        assert_eq!(flat.regime(), Some(Family::Flat));
        assert_eq!((flat.lifted, flat.flat), (1, 7));

        let mut clearing = LiftWitness::default();
        for k in 0..8 {
            clearing.observe(key(k as f64, 40.0, if k % 4 == 0 { 1.0 } else { 0.0 }));
        }
        assert_eq!(clearing.regime(), Some(Family::Lifted), "live most of the time");
        assert_eq!(clearing.cleared, 2, "and cleared when the flag is set");
    }

    #[test]
    fn merging_two_runs_adds_the_counts_and_keeps_the_worst() {
        let (mut a, mut b) = (LiftWitness::default(), LiftWitness::default());
        a.observe(key(3.0, 10.0, 0.0));
        b.observe(key(3.0, 1e9, 0.0));
        let worst = b.worst;
        a.merge(&b);
        assert_eq!((a.total, a.marked), (2, 2));
        assert_eq!(a.worst, worst, "the later position carries the larger offset");
    }
}

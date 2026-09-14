//! The point at which float64 stops being able to hold the answer.
//!
//! `todo3.md` section 4: the construction's exactness is exactness of integers
//! below `2^53`, and the score `2qk - k^2` of a parabolic key crosses that at
//! `k = ceil(sqrt(2^53)) = 94 906 266`.  Neither reference runtime says so and
//! neither checks; above the wall the answer is not slow or approximate, it is
//! wrong, and it is wrong silently.
//!
//! The condition is `S.delta < 1` of `ALM.FloatGrid.fp_eval_exact_of_grid`:
//! near the winning score, the representable doubles must still be closer
//! together than the gap between the winner and the runner-up.  Stated that way
//! it is scale-invariant, and it has to be, because the real machine does not
//! query at unit scale.

/// `2^53`: the last magnitude at which consecutive doubles are one apart.
pub const GRID_LIMIT: f64 = 9007199254740992.0;

/// The distance from `x` to the next double away from zero.
pub fn ulp(x: f64) -> f64 {
    let m = x.abs();
    if !m.is_finite() {
        return f64::INFINITY;
    }
    f64::from_bits(m.to_bits() + 1) - m
}

/// Whether a winning score of this size can still be told apart from a runner-up
/// `margin` below it.
///
/// An absolute test — "is the score past `2^53`" — is the wrong one, and the
/// real model shows why.  Its queries are the parabolic `(q, 1)` multiplied by
/// a hard-attention scale of `sqrt(2) * 10^10`, so the winning score passes
/// `2^53` at **position 799** of a 1034-token `hello` run while the answer
/// stays exactly right for another million tokens.  The scale multiplies the
/// score and the gap between the two best keys by very nearly the same factor,
/// so an absolute test is all false alarm.  What matters is whether
/// `ulp(score) <= margin`, and for parabolic integer keys the margin is `|qy|`,
/// the query's own second coordinate.  At `qy = 1` this reduces to the familiar
/// `score < 2^53`.
///
/// "Very nearly" is the interesting part.  `ulp` is relative and the margin is
/// not, so the cancellation is exact only up to where the scale sits inside its
/// binade, and the wall moves by up to half of one:
///
/// ```text
/// query scale            first q whose winner is unseparable
/// 1                                 94 906 266   (= ceil sqrt 2^53)
/// sqrt(2) * 10^10                   73 966 031   (22 % earlier)
/// ```
///
/// `todo3.md` section 4 states the first number.  The shipped model runs at the
/// second scale, so the second number is the one that bounds it.
///
/// A margin of zero is not a crossing.  It means the query is the zero vector,
/// every key scores `0`, and the tie is exact — resolving it is the tie-break's
/// job and float64 does it perfectly.
pub fn off_the_grid(score: f64, margin: f64) -> bool {
    margin != 0.0 && ulp(score) > margin.abs()
}

/// How much of the available separation a query used: `ulp(score) / |margin|`.
///
/// This is the quantity `off_the_grid` thresholds at `1`, and reporting it
/// rather than the threshold is what makes the cost of the query scale
/// visible.  A run whose worst ratio is `0.5` is one binade from the wall; a
/// run whose worst ratio is `18` answered a query it could not see.
pub fn grid_ratio(score: f64, margin: f64) -> f64 {
    if margin == 0.0 {
        0.0
    } else {
        ulp(score) / margin.abs()
    }
}

/// One query that float64 could not answer, kept for the report.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Crossing {
    pub score: f64,
    pub query: [f64; 2],
    /// The winning key, where the code path knows it: the `qy == 0` shortcut
    /// reads an aggregate at one end of the envelope and never forms one.
    pub key: Option<[f64; 2]>,
}

/// A head's record of whether it has answered a query float64 could not
/// separate, and the first one at which that happened.
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct GridWitness {
    pub first: Option<Crossing>,
    /// How many queries were answered off the grid.
    pub count: usize,
    /// The largest `grid_ratio` seen, crossing or not: how close the run came.
    pub worst: f64,
    /// The query that attained it.
    pub worst_at: Option<Crossing>,
}

impl GridWitness {
    /// Record a winning score against the margin it had to beat, returning
    /// whether the pair was past what float64 can separate.
    pub fn observe(&mut self, score: f64, margin: f64, query: [f64; 2], key: Option<[f64; 2]>) -> bool {
        let r = grid_ratio(score, margin);
        if r > self.worst {
            self.worst = r;
            self.worst_at = Some(Crossing { score, query, key });
        }
        if off_the_grid(score, margin) {
            self.count += 1;
            self.first.get_or_insert(Crossing { score, query, key });
            true
        } else {
            false
        }
    }

    pub fn merge(&mut self, other: &GridWitness) {
        if self.first.is_none() {
            self.first = other.first;
        }
        self.count += other.count;
        if other.worst > self.worst {
            self.worst = other.worst;
            self.worst_at = other.worst_at;
        }
    }

    pub fn is_clean(&self) -> bool {
        self.count == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ulp_is_the_gap_to_the_next_double() {
        assert_eq!(ulp(1.0), f64::EPSILON);
        assert_eq!(ulp(GRID_LIMIT), 2.0);
        assert_eq!(ulp(GRID_LIMIT / 2.0), 1.0);
    }

    #[test]
    fn at_unit_scale_the_wall_is_the_measured_one() {
        // A query answered by its own key scores `q^2`, against a margin of 1.
        let hit = |q: f64| off_the_grid(q * q, 1.0);
        assert!(!hit(94906265.0));
        assert!(hit(94906266.0));
    }

    #[test]
    fn the_hard_attention_scale_nearly_cancels_and_costs_22_percent() {
        // The real model's query at position 799, taken from a `hello` run:
        // the score is past 2^53 and the answer is nonetheless exact, because
        // the runner-up is a whole `qy` away rather than a whole 1.
        let qy = 14142135623.730951;
        let score = 9028359010593056.0;
        assert!(score > GRID_LIMIT, "an absolute test would fire here");
        assert!(!off_the_grid(score, qy), "and it would be wrong to");

        // Where it does fire is 22 % before the unit-scale wall, because `ulp`
        // is relative and the margin is not.
        let scaled = |q: f64| off_the_grid(qy * q * q, qy);
        assert!(!scaled(73966030.0));
        assert!(scaled(73966031.0));
        assert!(scaled(94906266.0), "past the unit wall it is certainly gone");
    }

    #[test]
    fn a_disabled_key_is_not_a_crossing() {
        // `clear_key` subtracts `BIG = 1e30` from a key's second coordinate
        // (`graph/core.py:8,314`).  Such a key is nowhere near winning, so it
        // never reaches the test at all — what is tested is the winner.
        let mut w = GridWitness::default();
        assert!(!w.observe(4.0 * 94906.0, 1.0, [1.0, 1.0], None));
        assert!(w.is_clean());
    }

    #[test]
    fn a_zero_query_ties_everything_exactly_and_is_not_a_crossing() {
        assert!(!off_the_grid(0.0, 0.0));
        assert_eq!(grid_ratio(0.0, 0.0), 0.0);
    }

    #[test]
    fn the_scale_shuffles_binades_and_only_the_threshold_moves() {
        // Both pairs are the worst query of a real run, at the two scales:
        // `hello` and `addition`, against the same value-keyed head.  The
        // scale helps one and hurts the other, by the two factors it can
        // apply -- `2^33 / s` and `2^34 / s` -- so it is not a uniform loss.
        let s = 14142135623.730951;
        let key = [673720322.0, -1.1347476806894592e17];
        let ratio = |qx: f64, qy: f64| grid_ratio(qx * key[0] + qy * key[1], qy);

        let hello = (ratio(4294967380.0, 1.0), ratio(6.074001118746039e19, s));
        assert_eq!(hello.0, 512.0);
        assert!(hello.1 < hello.0, "the scale helped here: {hello:?}");

        let addition = (ratio(8589934503.0, 1.0), ratio(1.2148001874039192e20, s));
        assert_eq!(addition.0, 1024.0);
        assert!(addition.1 > addition.0, "and hurt here: {addition:?}");

        // What the scale does move, one way, is the threshold.  Across the
        // whole binade `[2^52, 2^53)` the unit-scale ratio is exactly `1` --
        // separable, to the last unit -- and the scale pushes the upper 72 %
        // of it past.  That is the 22 % of section 4a, in `q` rather than in
        // `q^2`.
        let x = 1.3 * (GRID_LIMIT / 2.0);
        assert!(!off_the_grid(x, 1.0));
        assert!(off_the_grid(s * x, s));
    }

    #[test]
    fn a_witness_keeps_the_first_crossing_and_counts_the_rest() {
        let mut w = GridWitness::default();
        assert!(w.is_clean());
        assert!(!w.observe(1.0, 1.0, [1.0, 1.0], None));
        assert!(w.observe(GRID_LIMIT, 1.0, [2.0, 3.0], Some([4.0, 5.0])));
        assert!(w.observe(-2.0 * GRID_LIMIT, 1.0, [6.0, 7.0], None));
        let first = w.first.expect("a crossing was recorded");
        assert_eq!(first.score, GRID_LIMIT);
        assert_eq!(first.query, [2.0, 3.0]);
        assert_eq!(first.key, Some([4.0, 5.0]));
        assert_eq!(w.count, 2);
    }
}

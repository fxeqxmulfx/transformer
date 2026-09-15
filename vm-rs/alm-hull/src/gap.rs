//! How far the winner of a query is from the runner-up, in key steps.
//!
//! `todo3.md` section 2a.  Two writes to one logical key should reach a head
//! with the same key and tie, which is what `HullMeta::last_seq` is there to
//! resolve.  They do not: the key is a matvec through the residual stream and
//! the matvec rounds, so the two keys differ in the last bits and there is no
//! tie to break.  What holds the ordering up instead is the compiler's
//! `LATEST_ALPHA * inv_log_pos(p)`, a term added to `ky` that separates the
//! writes by far more than the rounding does.
//!
//! That makes the gap between the best and second-best score the diagnostic
//! quantity: measured in units of `|qy|`, one step of the key, a gap near `1`
//! is two genuinely different keys, a gap near `10^-5` is the perturbation
//! doing its job, and a gap near `10^-15` is rounding deciding the answer.
//! Only the brute head can report it — it scores everything anyway, while the
//! hull visits the winner and its ties and stops.
//!
//! And it is the gap the guard should be read against.  `grid.rs` is handed
//! `|qy|` as the margin — one key step, what two distinct integer keys would
//! be worth — while `ALM.GuardSep.cmp_of_guard` asks for
//! `hsep : sigma <= |a - b|`, the separation the two scores *actually* have.
//! The assumed margin is neither an upper nor a lower bound on that.  Where
//! the query lands between two keys and the recency offset decides the
//! winner, the real gap is a millionth of a step and the assumption is far too
//! generous; where the keys are byte patterns thousands apart, it is too mean
//! by the square of their separation, and the guard fires on a query that was
//! never in doubt.  Both were measured, on the same run.
//!
//! The other half of `cmp_of_guard` is `hd1, hd2 : |a' - a| <= ulpOf p E / 2`:
//! each score within half an ulp of exact, which is what one rounding costs.
//! The runtime does not spend one rounding.  It computes `q0*k0 + q1*k1`, two
//! products and a sum, and on a parabolic key the two products are near
//! `2k^2` and `-k^2` while their sum is near `k^2` — so the terms are three
//! times the result and their rounding error is carried into it undiminished.
//! `score_error_bound` is that bound, `u(2 + u)(|q0*k0| + |q1*k1|)` of
//! `ALM.DotError.dot_error_le`, `cmp_of_dot_guard` is what clearing twice it
//! decides, and
//! `unresolved` counts the queries whose real gap did not clear twice it —
//! twice because both the winner's score and the runner-up's carry it.
//! Measuring against `ulp(best)` instead would understate the error by the
//! cancellation factor and could only ever fail at a binade boundary, where
//! the predecessor is half an ulp away — which is a fact about `ulp` and not
//! about the model.
//!
//! `ALM.ScoreGap` is the gap as a number: `keyGap_scale_free` is why dividing
//! by the step makes it a property of the keys and not of the query scale, and
//! `one_le_keyGap_iff` is what a gap of `1` means — exactly the separation
//! `ALM.GuardSep.cmp_of_guard` needs, so a run whose worst gap is at least `1`
//! and whose `GridWitness` is clean answered every query and guessed none.
//! The `step` divided out here is the same number `grid.rs` is handed as the
//! `margin` (`ALM.QueryScale.gridScale`), which is what lets the two reports be
//! read against each other.

/// A gap this small, in key steps, is smaller than any `LATEST_ALPHA` the
/// compiler would emit over a reachable position, so it is the key path's own
/// rounding and not a separation anyone intended.
pub const NOISE: f64 = 1e-9;

/// How far `fl(q0*k0 + q1*k1)` can sit from the exact value, given that the
/// two products summed to `terms` in absolute value.
///
/// `ALM.DotError.dot_error_le`: each product is within `u = eps / 2`
/// relative, the sum within `u` relative, and the sum's magnitude is at most
/// `(1 + u) * terms` once the products have grown, so the whole is at most
/// `u * (2 + u) * terms`.  What makes it worth computing rather than
/// assuming is the cancellation: `ALM.DotError.dotTerms_markKey_self` says a parabolic key
/// queried near itself forms `2k^2` and `-k^2` and answers `k^2`, so `terms`
/// is three times the result and the bound is three times what `ulp(result)`
/// would suggest.
pub fn score_error_bound(terms: f64) -> f64 {
    // `u * (2 + u)` is not representable: with `u = eps / 2`, `2 + u` rounds
    // to `2` and the product would land a hair *below* the theorem's bound.
    // Two `next_up`s put it back above, with room for the `u^2` term and for
    // this multiplication's own rounding alike -- one ulp is `eps` relative
    // and the shortfall is `eps^2`.
    (f64::EPSILON * terms.abs()).next_up().next_up()
}

/// The gaps a run's queries showed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScoreGaps {
    /// Queries with a strictly-lower runner-up, so with a gap to report.
    pub total: usize,
    /// How many of those had a gap under `NOISE` key steps.
    pub noise: usize,
    /// The smallest gap seen, in key steps.  `INFINITY` until one is.
    pub worst: f64,
    /// The query that attained it.
    pub worst_at: Option<[f64; 2]>,
    /// Queries whose real gap did not clear twice the error the dot product
    /// can carry: `best - second <= 2 * score_error_bound(terms)`.  This is
    /// `ALM.GuardSep.cmp_of_guard` failing on the model the runtime actually
    /// computes with, and it is the case where the computed order can differ
    /// from the exact one.
    pub unresolved: usize,
    /// The largest `2 * score_error_bound(terms) / (best - second)` seen: how
    /// much of the real separation a query used, where `grid.rs`'s
    /// `grid_ratio` reports how much of the assumed one it used.
    pub worst_guard: f64,
    /// The query that attained it.
    pub worst_guard_at: Option<[f64; 2]>,
}

impl Default for ScoreGaps {
    fn default() -> Self {
        ScoreGaps {
            total: 0,
            noise: 0,
            worst: f64::INFINITY,
            worst_at: None,
            unresolved: 0,
            worst_guard: 0.0,
            worst_guard_at: None,
        }
    }
}

impl ScoreGaps {
    /// `best` is the winning score, `second` the largest strictly below it,
    /// and `terms` the largest `|q0*k0| + |q1*k1|` any entry of the head
    /// formed at this query — the size the rounding of the dot product is
    /// relative to.  Pass `best.abs()` for a score computed without
    /// cancellation, which is the bound `score_error_bound` degenerates to.
    pub fn observe(&mut self, best: f64, second: f64, q: [f64; 2], terms: f64) {
        if !second.is_finite() {
            return;
        }
        let step = if q[1] != 0.0 { q[1].abs() } else { q[0].abs() };
        if step == 0.0 || !step.is_finite() {
            return;
        }
        let gap = (best - second) / step;
        if !gap.is_finite() || gap <= 0.0 {
            return;
        }
        self.total += 1;
        if gap < NOISE {
            self.noise += 1;
        }
        if gap < self.worst {
            self.worst = gap;
            self.worst_at = Some(q);
        }
        // The guard against the separation this query really had, on the
        // error the dot product really carries.  `best` and `second` are
        // within a factor of two of each other whenever this is close, so the
        // subtraction is exact where the answer matters.
        let guard = 2.0 * score_error_bound(terms) / (best - second);
        if guard >= 1.0 {
            self.unresolved += 1;
        }
        if guard > self.worst_guard {
            self.worst_guard = guard;
            self.worst_guard_at = Some(q);
        }
    }

    pub fn merge(&mut self, other: &ScoreGaps) {
        self.total += other.total;
        self.noise += other.noise;
        if other.worst < self.worst {
            self.worst = other.worst;
            self.worst_at = other.worst_at;
        }
        self.unresolved += other.unresolved;
        if other.worst_guard > self.worst_guard {
            self.worst_guard = other.worst_guard;
            self.worst_guard_at = other.worst_guard_at;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_distinct_integer_keys_are_a_whole_step_apart() {
        // The parabolic embedding: querying `k` exactly, the neighbour `k + 1`
        // scores one key step lower, whatever the scale.
        let mut w = ScoreGaps::default();
        for s in [1.0, 14142135623.730951] {
            let (k, q) = (7.0f64, [7.0 * s, s]);
            let score = |x: f64| q[0] * (2.0 * x) + q[1] * (-x * x);
            let terms = |x: f64| (q[0] * 2.0 * x).abs() + (q[1] * x * x).abs();
            w.observe(score(k), score(k + 1.0), q, terms(k).max(terms(k + 1.0)));
        }
        assert_eq!((w.total, w.noise), (2, 0));
        assert!((w.worst - 1.0).abs() < 1e-9, "one step, at either scale: {w:?}");
    }

    #[test]
    fn the_perturbation_separates_and_the_rounding_does_not() {
        // Two writes to the same key, as the model presents them: with
        // `LATEST_ALPHA * inv_log_pos` added to `ky`, and without it, where
        // only the matvec's last bits separate them.
        let q = [7.0, 1.0];
        let ky = -49.0;
        let score = |y: f64| q[0] * 14.0 + q[1] * y;
        let terms = |y: f64| (q[0] * 14.0).abs() + (q[1] * y).abs();

        let mut perturbed = ScoreGaps::default();
        perturbed.observe(score(ky + 0.3 * 1.0e-5), score(ky), q, terms(ky));
        assert_eq!(perturbed.noise, 0, "the compiler's separation is visible");

        let mut rounded = ScoreGaps::default();
        let nudged = ky + 8.0 * f64::EPSILON * ky.abs();
        rounded.observe(score(nudged), score(ky), q, terms(ky));
        assert_eq!(rounded.noise, 1, "the matvec's is not");
        assert!(rounded.worst < NOISE);
    }

    #[test]
    fn the_step_is_the_margin_the_guard_is_handed() {
        // `ALM.ScoreGap.one_le_keyGap_iff`: a gap of at least one key step is
        // `margin <= best - second`, the hypothesis `cmp_of_guard` needs.  It
        // is a hypothesis about the same number the guard thresholds against,
        // and this is the line that says so — both branches of it, the second
        // being `head.rs`'s `qy == 0` shortcut.
        let queries: [[f64; 2]; 3] =
            [[7.0, 1.0], [7.0 * 14142135623.730951, 14142135623.730951], [5.0, 0.0]];
        for q in queries {
            let step = if q[1] != 0.0 { q[1].abs() } else { q[0].abs() };
            let (best, second) = (1.0e6 + step, 1.0e6);

            let mut w = ScoreGaps::default();
            w.observe(best, second, q, 3.0e6);
            assert_eq!(w.worst, 1.0, "one step, whatever the scale: {q:?}");
            assert!(step <= best - second, "which is the guard's hypothesis");
            assert!(!crate::grid::off_the_grid(best, step), "and the guard passes");
        }
    }

    #[test]
    fn the_bound_is_the_one_the_theorem_proves() {
        // `ALM.DotError.dot_error_le` bounds the error at `u * (2 + u) * T`,
        // which is above the first-order `2 * u * T` by the `u^2` the sum
        // spends on products that have already grown.
        let t = 3.0e17;
        assert!(score_error_bound(t) > f64::EPSILON * t, "above the first-order bound");
        assert!(score_error_bound(t) < f64::EPSILON * t * (1.0 + 1.0e-15), "and by two ulp");
        assert_eq!(score_error_bound(-t), score_error_bound(t), "a bound, so unsigned");
    }

    #[test]
    fn the_cancellation_is_what_the_error_bound_is_for() {
        // A score formed without cancellation costs about one ulp.
        assert!(score_error_bound(1e17) <= 2.0 * crate::grid::ulp(1e17));
        // The same score formed as `2k^2 - k^2` costs three times that,
        // and `ulp(result)` cannot see the difference.
        let k2 = 1e17;
        assert!(score_error_bound(3.0 * k2) >= 2.5 * crate::grid::ulp(k2));
    }

    #[test]
    fn the_guard_is_read_against_the_real_gap_and_the_real_error() {
        // A gap of one against a score of `1e6`: the error is `2*eps*3e6`,
        // about `1.3e-9` of the gap, and nothing is close to unresolved.
        let mut w = ScoreGaps::default();
        w.observe(1e6, 1e6 - 1.0, [1.0, 1.0], 3e6);
        assert_eq!(w.unresolved, 0);
        assert!(w.worst_guard < 1e-8);

        // And two that are adjacent doubles: the ulp reading calls this the
        // boundary case and lets it through, while the error the dot product
        // really carries is three ulp, so the order can invert.
        let big = 1e17;
        let mut b = ScoreGaps::default();
        b.observe(big + crate::grid::ulp(big), big, [1.0, 1.0], 3.0 * big);
        assert_eq!(b.unresolved, 1);
        assert!(b.worst_guard > 1.0);
    }

    #[test]
    fn the_assumed_margin_is_neither_a_bound_nor_the_other_bound() {
        // Where the query sits between two keys and the recency offset
        // decides, the real gap is a fraction of a step and the assumed
        // margin of one step is far too generous.
        let step = 1.0;
        let (best, second) = (1e6, 1e6 - 1e-6);
        let mut tight = ScoreGaps::default();
        tight.observe(best, second, [0.0, step], 3e6);
        assert!(tight.worst < 1e-5 && tight.worst > 0.0);
        assert!(step > best - second, "the assumed margin overstates the real one");
        assert!(!crate::grid::off_the_grid(best, step), "so the assumed guard passes");

        // And where the keys are far apart the assumed margin understates it,
        // so the assumed guard fires on a query that was never in doubt.
        let (best, second) = (1e17, 1e17 - 1e9);
        let mut loose = ScoreGaps::default();
        loose.observe(best, second, [0.0, step], 3e17);
        assert_eq!(loose.unresolved, 0);
        assert!(loose.worst_guard < 1e-4, "resolved with four orders to spare");
        assert!(crate::grid::off_the_grid(best, step), "and the assumed guard fires anyway");
    }

    #[test]
    fn an_exact_tie_is_not_a_gap() {
        let mut w = ScoreGaps::default();
        w.observe(12.0, 12.0, [3.0, 1.0], 12.0);
        assert_eq!(w.total, 0, "a tie is `last_seq`'s business, not this one's");
        assert_eq!(w.worst, f64::INFINITY);
    }
}

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

/// A gap this small, in key steps, is smaller than any `LATEST_ALPHA` the
/// compiler would emit over a reachable position, so it is the key path's own
/// rounding and not a separation anyone intended.
pub const NOISE: f64 = 1e-9;

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
}

impl Default for ScoreGaps {
    fn default() -> Self {
        ScoreGaps { total: 0, noise: 0, worst: f64::INFINITY, worst_at: None }
    }
}

impl ScoreGaps {
    /// `best` is the winning score, `second` the largest strictly below it.
    pub fn observe(&mut self, best: f64, second: f64, q: [f64; 2]) {
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
    }

    pub fn merge(&mut self, other: &ScoreGaps) {
        self.total += other.total;
        self.noise += other.noise;
        if other.worst < self.worst {
            self.worst = other.worst;
            self.worst_at = other.worst_at;
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
            w.observe(score(k), score(k + 1.0), q);
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

        let mut perturbed = ScoreGaps::default();
        perturbed.observe(score(ky + 0.3 * 1.0e-5), score(ky), q);
        assert_eq!(perturbed.noise, 0, "the compiler's separation is visible");

        let mut rounded = ScoreGaps::default();
        let nudged = ky + 8.0 * f64::EPSILON * ky.abs();
        rounded.observe(score(nudged), score(ky), q);
        assert_eq!(rounded.noise, 1, "the matvec's is not");
        assert!(rounded.worst < NOISE);
    }

    #[test]
    fn an_exact_tie_is_not_a_gap() {
        let mut w = ScoreGaps::default();
        w.observe(12.0, 12.0, [3.0, 1.0]);
        assert_eq!(w.total, 0, "a tie is `last_seq`'s business, not this one's");
        assert_eq!(w.worst, f64::INFINITY);
    }
}

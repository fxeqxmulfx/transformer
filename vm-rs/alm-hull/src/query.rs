//! Whether the queries the machine asks are the integers the exactness
//! argument assumes they are.
//!
//! `todo3.md` section 8: the hypothesis `q : Z` of
//! `ALM.FloatGrid.fp_exact_of_grid` is a hypothesis about the *query*, and the
//! released machine forms several of its queries by a route that cannot
//! preserve it.  A uniform-attention head returns `vsum * (1.0 / count)` and
//! the FFN multiplies by `position` to recover the sum, so the value that
//! reaches the next query is `fl(fl(s * fl(1/p)) * p)`, which is `s` only 74 %
//! of the time; `cursor` is one of those values and goes into a lookup as
//! `5 * cursor + 1`.
//!
//! That is read off the source.  What it costs is a measurement, and this is
//! the instrument: every query, divided by its own `|qy|` so that the scale is
//! gone, against the nearest integer, in ulps of itself.

use crate::grid::ulp;

/// What a run's queries looked like once the scale was divided out.
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct IntegerQueries {
    /// Queries examined: those with `qy != 0`, where `qx / |qy|` is the value
    /// being looked up.  The `qy == 0` shortcut asks for an extreme of the
    /// envelope rather than for a value, and is not counted.
    pub total: usize,
    /// How many of those were not integers.
    pub off: usize,
    /// The largest distance to the nearest integer.  Absolute, not relative:
    /// the keys are one apart, so this is the quantity that has to stay under
    /// `1/2` for the argmax to be the intended one, and
    /// `ALM.FloatHull.cmp_of_sep` wants it as `delta_1 + delta_2 < |a - b|`.
    pub worst: f64,
    /// The query that attained it, and that offset in ulps of it — the form
    /// `todo3.md` section 8 states its margin in.
    pub worst_at: Option<(f64, f64)>,
}

impl IntegerQueries {
    pub fn observe(&mut self, q: [f64; 2]) {
        if q[1] == 0.0 {
            return;
        }
        let t = q[0] / q[1].abs();
        if !t.is_finite() {
            return;
        }
        self.total += 1;
        let d = (t - t.round()).abs();
        if d == 0.0 {
            return;
        }
        self.off += 1;
        if d > self.worst {
            self.worst = d;
            self.worst_at = Some((t, d / ulp(t)));
        }
    }

    pub fn merge(&mut self, other: &IntegerQueries) {
        self.total += other.total;
        self.off += other.off;
        if other.worst > self.worst {
            self.worst = other.worst;
            self.worst_at = other.worst_at;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_integer_query_at_any_scale_is_an_integer_query() {
        let mut w = IntegerQueries::default();
        let s = 14142135623.730951;
        for q in [1.0, 336860161.0, 94906265.0] {
            w.observe([s * q, s]);
        }
        assert_eq!((w.total, w.off), (3, 0), "{w:?}");
    }

    #[test]
    fn the_shortcut_path_asks_for_a_direction_and_is_not_counted() {
        let mut w = IntegerQueries::default();
        w.observe([1.5, 0.0]);
        assert_eq!(w.total, 0);
    }

    #[test]
    fn the_round_trip_through_an_average_is_what_it_catches() {
        // `todo3.md` section 8's own first witness: a sum of 3 over 5
        // positions, averaged and multiplied back, is 3.0000000000000004.
        let (s, p) = (3.0f64, 5.0f64);
        let round_trip = (s * (1.0 / p)) * p;
        assert_ne!(round_trip, s);

        let mut w = IntegerQueries::default();
        w.observe([round_trip, 1.0]);
        assert_eq!((w.total, w.off), (1, 1));
        let (at, ulps) = w.worst_at.expect("an offset was recorded");
        assert_eq!(at, round_trip);
        assert_eq!(ulps, 1.0, "one ulp, which is the whole margin");
        assert!(w.worst < 0.5, "and far inside the half-unit `cmp_of_sep` needs");
    }
}

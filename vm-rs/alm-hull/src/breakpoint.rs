//! Hull breakpoints as exact rationals.
//!
//! A line `y = m x + b` stops being the maximum at the `x` where it meets its
//! successor, `x = (b' - b) / (m - m')`.  The C++ original evaluates that
//! quotient in `long double` and stores the rounded result; every later
//! comparison then works on a rounded number.  Here the numerator and the
//! denominator are kept apart and comparisons cross-multiply, so the order is
//! the exact order of the real breakpoints.  `ALM.FloatHull.cmp_of_sep` is the
//! statement this removes the hypothesis of: with `delta_1 = delta_2 = 0` the
//! comparison of the perturbed values is the comparison of the true ones.

use crate::exact::cross_sign;
use core::cmp::Ordering;

/// The `x` at which one line gives way to the next, or an end of the line.
///
/// `Finite { num, den }` denotes `num / den` with `den > 0`; the representation
/// is not normalised, since nothing here needs a canonical form.  `Unset` is
/// the value a freshly inserted line carries before its breakpoint has been
/// computed, and orders below everything so that it can never be mistaken for
/// a real one.
#[derive(Clone, Copy, Debug)]
pub enum Break {
    Unset,
    NegInf,
    Finite { num: f64, den: f64 },
    PosInf,
}

impl Break {
    /// The breakpoint of the line `(mx, bx)` against the line `(my, by)`.
    ///
    /// Equal slopes never meet: the one with the larger intercept wins
    /// everywhere, which is `PosInf`, and otherwise nowhere, which is `NegInf`.
    pub fn between(mx: f64, bx: f64, my: f64, by: f64) -> Break {
        if mx == my {
            return if bx >= by { Break::PosInf } else { Break::NegInf };
        }
        let num = by - bx;
        let den = mx - my;
        if den > 0.0 {
            Break::Finite { num, den }
        } else {
            Break::Finite { num: -num, den: -den }
        }
    }

    /// The query point `qx / qy` as a breakpoint, for `qy != 0`.
    pub fn ratio(qx: f64, qy: f64) -> Break {
        debug_assert!(qy != 0.0);
        if qy > 0.0 {
            Break::Finite { num: qx, den: qy }
        } else {
            Break::Finite { num: -qx, den: -qy }
        }
    }

    fn rank(self) -> i32 {
        match self {
            Break::Unset => 0,
            Break::NegInf => 1,
            Break::Finite { .. } => 2,
            Break::PosInf => 3,
        }
    }
}

impl PartialEq for Break {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Break {}

impl PartialOrd for Break {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Break {
    fn cmp(&self, other: &Self) -> Ordering {
        match (*self, *other) {
            (Break::Finite { num: n1, den: d1 }, Break::Finite { num: n2, den: d2 }) => {
                cross_sign(n1, d2, n2, d1)
            }
            (a, b) => a.rank().cmp(&b.rank()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_order_is_the_order_of_the_quotients() {
        let a = Break::Finite { num: 1.0, den: 3.0 };
        let b = Break::Finite { num: 2.0, den: 6.0 };
        let c = Break::Finite { num: 1.0, den: 2.0 };
        assert_eq!(a, b);
        assert!(a < c);
        assert!(Break::NegInf < a && a < Break::PosInf);
        assert!(Break::Unset < Break::NegInf);
    }

    #[test]
    fn a_negative_denominator_is_folded_not_dropped() {
        // -1/-3 and 1/3 are the same point, and 1/-3 is its reflection.
        assert_eq!(Break::ratio(-1.0, -3.0), Break::ratio(1.0, 3.0));
        assert!(Break::ratio(1.0, -3.0) < Break::ratio(1.0, 3.0));
    }

    #[test]
    fn breakpoints_that_long_double_cannot_separate() {
        // (2^53+2)/2^53 and (2^53+4)/(2^53+2) cross-multiply to products four
        // apart at 2^106, where the ulp is 2^54.  Dividing first, as the C++
        // does, makes these one number.
        let x = Break::Finite { num: 9007199254740994.0, den: 9007199254740992.0 };
        let y = Break::Finite { num: 9007199254740996.0, den: 9007199254740994.0 };
        assert!(x > y);
        assert_ne!(x, y);
    }

    #[test]
    fn between_is_the_meeting_point() {
        // y = 0x + 1 and y = 1x + 0 meet at x = 1.
        assert_eq!(Break::between(0.0, 1.0, 1.0, 0.0), Break::Finite { num: 1.0, den: 1.0 });
        assert_eq!(Break::between(2.0, 5.0, 2.0, 4.0), Break::PosInf);
        assert_eq!(Break::between(2.0, 4.0, 2.0, 5.0), Break::NegInf);
    }
}

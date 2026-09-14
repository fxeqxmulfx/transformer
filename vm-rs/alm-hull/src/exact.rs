//! Error-free transformations and the exact sign of a short float expansion.
//!
//! The C++ original (`attention/hull2d_cht.h`) keeps every hull breakpoint in
//! `long double` and compares breakpoints by `>=`.  Rust has no 80-bit float,
//! so a transliteration would drop from a 64-bit significand to 53 bits exactly
//! where `ALM.FloatHull.fp_longDouble` asks for `u <= 2^-60`.  Instead of
//! losing precision we drop the division altogether: a breakpoint is kept as an
//! exact rational and compared by cross-multiplication, and the products are
//! compared with the routines below, which are exact for every finite input.
//!
//! Both transformations are standard (Dekker 1971; Knuth 1969); the expansion
//! sum is Shewchuk, "Adaptive Precision Floating-Point Arithmetic", 1997, §2.

/// `two_sum(a, b) = (s, e)` with `s = fl(a + b)` and `a + b = s + e` exactly.
#[inline]
pub fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    let e = (a - (s - bb)) + (b - bb);
    (s, e)
}

/// `two_prod(a, b) = (p, e)` with `p = fl(a * b)` and `a * b = p + e` exactly.
///
/// Uses a fused multiply-add, so the error term costs one instruction.
#[inline]
pub fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let e = a.mul_add(b, -p);
    (p, e)
}

/// The exact sign of `terms[0] + terms[1] + ... `, for up to eight finite terms.
///
/// The terms are accumulated into a non-overlapping expansion, whose sign is
/// the sign of its largest component.  No rounding occurs anywhere, so the
/// answer is the sign of the exact sum even when it cancels to zero.
pub fn expansion_sign(terms: &[f64]) -> core::cmp::Ordering {
    use core::cmp::Ordering;
    debug_assert!(terms.len() <= 8);

    let mut e = [0.0f64; 9];
    let mut len = 0usize;

    for &t in terms {
        let mut q = t;
        let mut out = 0usize;
        for i in 0..len {
            let (s, r) = two_sum(q, e[i]);
            if r != 0.0 {
                e[out] = r;
                out += 1;
            }
            q = s;
        }
        if q != 0.0 {
            e[out] = q;
            out += 1;
        }
        len = out;
    }

    if len == 0 {
        return Ordering::Equal;
    }
    e[len - 1].partial_cmp(&0.0).unwrap_or(Ordering::Equal)
}

/// The exact sign of `a * b - c * d`, for finite inputs.
///
/// This is the whole of the breakpoint comparison: `n1/d1 >= n2/d2` becomes
/// `n1 * d2 - n2 * d1 >= 0` once the denominators are known positive.
pub fn cross_sign(a: f64, b: f64, c: f64, d: f64) -> core::cmp::Ordering {
    let (p, pe) = two_prod(a, b);
    let (q, qe) = two_prod(c, d);
    expansion_sign(&[p, pe, -q, -qe])
}

/// The exact sign of `q . a - q . b`, for finite 2D vectors.
///
/// This makes a head return the true `argmax_k q . k` of the points it holds,
/// instead of the argmax of the rounded scores, so the hull head and the brute
/// one agree by construction rather than by luck.
///
/// It does **not** move the wall, and the reason is worth stating because it
/// is the reason no wider accumulator moves it either.  The wall of
/// `todo3.md` section 0 is not about how the score is computed: the key
/// coordinate is `-k^2`, and it stops being an exact `f64` at
/// `k = 94 906 267`.  Past that the two keys handed to this function are
/// already the wrong points.  Scanning both, the first query whose own key
/// stops being the strict argmax among its neighbours is `q = 94 906 266`
/// under rounded scoring and `q = 94 906 266` under exact scoring — the same
/// number, to the unit.  Exact arithmetic, `f128`, a `long double`
/// accumulator: none of them recover a coordinate that was rounded before it
/// arrived.  Only a wider embedding would, and that is a change to the
/// weights, not to the attention.
pub fn dot_cmp(q: [f64; 2], a: [f64; 2], b: [f64; 2]) -> core::cmp::Ordering {
    let (p0, e0) = two_prod(q[0], a[0]);
    let (p1, e1) = two_prod(q[1], a[1]);
    let (p2, e2) = two_prod(q[0], b[0]);
    let (p3, e3) = two_prod(q[1], b[1]);
    expansion_sign(&[p0, e0, p1, e1, -p2, -e2, -p3, -e3])
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::cmp::Ordering;

    #[test]
    fn two_prod_is_exact() {
        let a = 1.0 + f64::EPSILON;
        let (p, e) = two_prod(a, a);
        assert_ne!(e, 0.0, "the square of 1+eps does not fit in one double");
        assert_eq!(p + e, a * a + e);
    }

    #[test]
    fn cross_sign_separates_products_that_round_together() {
        // 2^53 * 2^53 and (2^53+2) * (2^53-2) differ by exactly 4, and the ulp
        // at 2^106 is 2^54, so both round to the same double.  A `long double`
        // has 64 bits of significand and cannot separate them either.
        let m = 9007199254740992.0; // 2^53
        let hi = 9007199254740994.0; // 2^53 + 2
        let lo = 9007199254740990.0; // 2^53 - 2
        assert_eq!(m * m, hi * lo, "the rounded products are equal");
        assert_eq!(cross_sign(m, m, hi, lo), Ordering::Greater);
        assert_eq!(cross_sign(hi, lo, m, m), Ordering::Less);
        assert_eq!(cross_sign(m, m, m, m), Ordering::Equal);
    }

    /// The parabolic embedding: key `k` at `(2k, -k^2)`, query `q` at `(q, 1)`.
    fn key(k: f64) -> [f64; 2] {
        [2.0 * k, -(k * k)]
    }

    #[test]
    fn dot_cmp_is_exact_on_the_points_it_is_given() {
        let q = 94906265.0;
        let score = |k: f64| 2.0 * q * k - k * k;
        // One below the wall every coordinate is still exact, and both the
        // rounded scores and the exact comparison put the query's own key first.
        assert!(score(q) > score(q - 1.0));
        assert_eq!(dot_cmp([q, 1.0], key(q), key(q - 1.0)), Ordering::Greater);
        assert_eq!(dot_cmp([q, 1.0], key(q), key(q)), Ordering::Equal);
    }

    #[test]
    fn the_wall_is_the_key_coordinate_not_the_arithmetic() {
        // `todo3.md` section 0 measures the first query that loses to a
        // neighbour at q = 94 906 266 = ceil(sqrt(2^53)).  Exact arithmetic on
        // the same stored points fails at exactly the same q, because what has
        // gone wrong is the coordinate, not the sum: -k^2 is rounded before
        // `dot_cmp` ever sees it.
        let q = 94906266i64;
        let qf = [q as f64, 1.0];

        // Left of the query, exact arithmetic still separates what f64 cannot.
        let lo = key((q - 1) as f64);
        assert_eq!(
            2.0 * q as f64 * (q - 1) as f64 - ((q - 1) * (q - 1)) as f64,
            2.0 * q as f64 * q as f64 - (q * q) as f64,
            "the rounded scores of k = q-1 and k = q are one double"
        );
        assert_eq!(dot_cmp(qf, key(q as f64), lo), Ordering::Greater);

        // Right of it, exact arithmetic ties, and that is the wall: the stored
        // coordinate of q+1 is 9 007 199 515 875 288, one short of the true
        // 9 007 199 515 875 289, and the missing unit is exactly the gap that
        // made q the argmax.
        let hi = key((q + 1) as f64);
        assert_eq!(-hi[1], 9007199515875288.0);
        assert_eq!((q + 1) * (q + 1), 9007199515875289);
        assert_eq!(
            dot_cmp(qf, key(q as f64), hi),
            Ordering::Equal,
            "exact arithmetic on rounded points still cannot tell them apart"
        );

        // Below the wall both agree, and agree with the truth.
        let p = (q - 1) as f64;
        let pf = [p, 1.0];
        assert_eq!(dot_cmp(pf, key(p), key(p + 1.0)), Ordering::Greater);
        assert_eq!(dot_cmp(pf, key(p), key(p - 1.0)), Ordering::Greater);
    }

    #[test]
    fn expansion_sign_cancels_exactly() {
        assert_eq!(expansion_sign(&[1e300, 1.0, -1e300]), Ordering::Greater);
        assert_eq!(expansion_sign(&[1e300, -1.0, -1e300]), Ordering::Less);
        assert_eq!(expansion_sign(&[1e300, 0.0, -1e300]), Ordering::Equal);
    }
}

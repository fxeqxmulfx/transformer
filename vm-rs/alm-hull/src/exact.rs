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

/// The exact sign of `terms[0] + terms[1] + ... `, for up to four finite terms.
///
/// The terms are accumulated into a non-overlapping expansion, whose sign is
/// the sign of its largest component.  No rounding occurs anywhere, so the
/// answer is the sign of the exact sum even when it cancels to zero.
pub fn expansion_sign(terms: &[f64]) -> core::cmp::Ordering {
    use core::cmp::Ordering;
    debug_assert!(terms.len() <= 4);

    let mut e = [0.0f64; 5];
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

    #[test]
    fn expansion_sign_cancels_exactly() {
        assert_eq!(expansion_sign(&[1e300, 1.0, -1e300]), Ordering::Greater);
        assert_eq!(expansion_sign(&[1e300, -1.0, -1e300]), Ordering::Less);
        assert_eq!(expansion_sign(&[1e300, 0.0, -1e300]), Ordering::Equal);
    }
}

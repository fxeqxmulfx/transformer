//! The constraint that does bite: the score squares the key.
//!
//! `alm_hull::grid` names the float64 wall -- `ceil(sqrt(2^53)) = 94 906 266`,
//! past which `2qk - k^2` no longer separates consecutive integer keys, and
//! the answer is silently wrong rather than slow or approximate.  The same
//! arithmetic gives a wall for every mantissa width, and the one that matters
//! for putting an exact head into an ordinary transformer is float32's.
//!
//! The test is `ulp(score) <= margin` with `margin = |qy| = 1` for parabolic
//! integer keys at unit query scale, and the worst score a key set of bound
//! `K` reaches is `K^2`.  So the wall is the smallest `K` with `K^2 >= 2^m`,
//! that is `ceil(2^(m/2))`:
//!
//! | format | mantissa | wall |
//! |---|---|---|
//! | float32 | 24 | 4 096 |
//! | bfloat16 | 8 | 16 |
//! | float64 | 53 | 94 906 266 |
//!
//! Four thousand addresses in float32 is the whole of the result.  It is not a
//! drift problem -- `drift` shows a float32 rounding of an address is three
//! orders inside the margin -- and no amount of care with the query touches
//! it.  It is the key's own magnitude, and the only way to shrink that is to
//! shrink the key: rebase to the current position, `KEY_OFFSET` in
//! `transformer_vm/graph/core.py:9`.

/// A binary floating-point format, by the width of its significand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Format {
    pub name: &'static str,
    /// Significand bits including the implicit one -- `IsBinary prec` of
    /// `ALM.FloatGrid`.
    pub mantissa: u32,
}

pub const BF16: Format = Format {
    name: "bfloat16",
    mantissa: 8,
};
pub const F16: Format = Format {
    name: "float16",
    mantissa: 11,
};
pub const F32: Format = Format {
    name: "float32",
    mantissa: 24,
};
pub const F64: Format = Format {
    name: "float64",
    mantissa: 53,
};

pub const FORMATS: [Format; 4] = [BF16, F16, F32, F64];

/// **The largest key bound whose scores a format still separates**, as the
/// smallest `K` with `K^2 >= 2^mantissa`.
pub fn score_wall(f: Format) -> u64 {
    let limit = (2.0f64).powi(f.mantissa as i32);
    let k = limit.sqrt().ceil() as u64;
    // `sqrt` rounds; step until the predicate is the one in the docstring.
    let mut k = k;
    while (k as f64) * (k as f64) < limit {
        k += 1;
    }
    while k > 1 && ((k - 1) as f64) * ((k - 1) as f64) >= limit {
        k -= 1;
    }
    k
}

/// `4096`.
pub fn wall_f32() -> u64 {
    score_wall(F32)
}

/// `94 906 266` -- `alm_hull::grid::GRID_LIMIT` read as a key bound.
pub fn wall_f64() -> u64 {
    score_wall(F64)
}

/// The score in float32, computed the way a head would.
#[inline]
pub fn score_f32(q: f32, k: f32) -> f32 {
    2.0 * k * q - k * k
}

/// **Where a float32 head actually starts returning the wrong key**, as the
/// smallest key bound at which some exact integer query fails to retrieve
/// itself out of `0..=bound`.
///
/// It lands one key above [`wall_f32`], and the reason is worth keeping.  The
/// derived wall is the smallest `K` with `K^2 >= 2^m`, which at `m = 24` is
/// the power of two `4096` itself -- and there `K^2 = 2^24` is still exactly
/// representable, as is the runner-up's `2^24 - 1`, because both sit on the
/// boundary rather than past it.  The first bound that genuinely loses a
/// comparison is `4097`.  So the derived wall is conservative, by exactly one
/// key, and only at a power of two.
pub fn first_failing_key_f32() -> u64 {
    let retrieves = |n: u64| -> bool {
        // `n - 1` and `n - 2` are the discriminating queries: the wall bites
        // where the winning score is itself near the format's limit.  An
        // exhaustive sweep of every query in `0..n` agrees with these five.
        let probes = [0u64, 1, n / 2, n - 2, n - 1];
        probes.iter().all(|&t| {
            let q = t as f32;
            let mut best = (f32::NEG_INFINITY, 0u64);
            for k in 0..n {
                let s = score_f32(q, k as f32);
                if s > best.0 {
                    best = (s, k);
                }
            }
            best.1 == t
        })
    };
    let (mut lo, mut hi) = (4u64, 1u64 << 14);
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if retrieves(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    hi - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_walls() {
        assert_eq!(wall_f32(), 4096);
        assert_eq!(wall_f64(), 94_906_266);
        assert_eq!(score_wall(BF16), 16);
    }

    /// `alm_hull::grid::GRID_LIMIT` is `2^53`, and the wall is its square root
    /// rounded up -- the two files agree without sharing a constant.
    #[test]
    fn agrees_with_the_hull_crate() {
        let limit = 9_007_199_254_740_992.0f64;
        assert_eq!(wall_f64(), limit.sqrt().ceil() as u64);
    }

    /// **The float32 wall, measured rather than derived** -- and one key above
    /// the derivation, for the reason `first_failing_key_f32` gives.
    #[test]
    fn measured_is_one_above_derived() {
        assert_eq!(first_failing_key_f32(), 4097);
        assert_eq!(first_failing_key_f32(), wall_f32() + 1);
    }

    /// Which is to say the head is exact on `0..=4096` and not on `0..=4097`.
    #[test]
    fn four_thousand_addresses_in_float32() {
        let retrieves = |bound: u64, q: u64| -> bool {
            let qf = q as f32;
            (0..=bound)
                .max_by(|&a, &b| {
                    score_f32(qf, a as f32)
                        .partial_cmp(&score_f32(qf, b as f32))
                        .unwrap()
                })
                .unwrap()
                == q
        };
        assert!((0..=4096).all(|q| retrieves(4096, q)));
        assert!(!(0..=4097).all(|q| retrieves(4097, q)));
    }
}

//! What it costs to write an address twice.
//!
//! An ALM has no store instruction: memory is the token sequence, and a write
//! is a position whose key is the address.  Writing an address a second time is
//! therefore two positions carrying the same key, and what orders them is
//! `graph/core.py:316`, one term added to the lifted key's second coordinate:
//!
//! ```text
//!     ky = -k^2 + LATEST_ALPHA * inv_log_pos(p)
//! ```
//!
//! with `LATEST_ALPHA = 0.3` and `inv_log_pos(p) = 1/ln 2 - 1/ln(p + 2)`
//! (`evaluator.py:230`).  That is the whole of rewriting, and `clear_key`'s
//! `BIG` is the whole of deletion.
//!
//! The term is added to `-k^2`, so it has to survive the ulp of a number of
//! size `a^2`.  Two consequences, and neither is written down anywhere.
//!
//! **The address space and the rewrite depth come out of one budget.**  The
//! recency term spans `[0, LATEST_ALPHA / ln 2)`, about `0.4328`, and a
//! rewrite is visible only while it exceeds `ulp(a^2)`.  So
//!
//! ```text
//!     addresses^2 * rewrites  <=  0.4328 * 2^(m - 1)
//! ```
//!
//! -- `3.6e6` in float32, `1.95e15` in float64.  float32 buys 1905 addresses
//! rewritten once, or 60 addresses rewritten a thousand times, and not both.
//! [`capacity`] is that constant and [`levels`] the count at one address.
//!
//! **And `inv_log_pos` saturates, so the real figure is far below that.**  The
//! bound above assumes the whole span is available, which would need the writes
//! spaced to use it.  Consecutive positions are what actually happens, and
//! there the increment is `LATEST_ALPHA / ((p + 2) ln^2(p + 2))`, which decays
//! while `ulp(a^2)` does not.  [`last_resolving_position`] measures it: at
//! address `10^5` in float64 a rewrite stops outranking its predecessor after
//! position **2552**, ninety times below the `227 000` the budget allows.
//! [`linear_capacity`] is the term that would not do that -- uniform increments
//! `span / p_max` -- and it reaches the budget exactly.

use crate::ceiling::Format;

/// `LATEST_ALPHA`, `graph/core.py:170`.
pub const RECENCY_ALPHA: f64 = 0.3;

/// `inv_log_pos`, `evaluator.py:230`.  Monotone, and bounded by `1 / ln 2`.
#[inline]
pub fn inv_log_pos(p: u64) -> f64 {
    1.0 / std::f64::consts::LN_2 - 1.0 / ((p + 2) as f64).ln()
}

/// The whole range the recency term can move a key through: `0.4328`.
///
/// It has to stay below the unit gap between distinct integer keys, which is
/// what makes `0.3` a safe choice and not a tuned one -- with the drift margin
/// of `crate::drift`, the condition on a query drift `d` is
/// `2|d| + span < 1`, so `|d| < 0.284`.
pub fn recency_span() -> f64 {
    RECENCY_ALPHA / std::f64::consts::LN_2
}

/// The distance to the next representable number above `x` in a format of this
/// mantissa width.  `2^(floor(log2 x) - m + 1)`.
pub fn ulp_at(x: f64, mantissa: u32) -> f64 {
    if x == 0.0 || !x.is_finite() {
        return f64::INFINITY;
    }
    let e = x.abs().log2().floor() as i32;
    (2.0f64).powi(e - mantissa as i32 + 1)
}

/// **How many times one address can be rewritten and still be told apart**, if
/// the writes were spaced to use the whole recency range.  An upper bound;
/// [`last_resolving_position`] is what the shipped term delivers.
pub fn levels(f: Format, addr: u64) -> f64 {
    let a = addr as f64;
    recency_span() / ulp_at(a * a, f.mantissa)
}

/// **`addresses^2 * rewrites`**, the budget the two share.
pub fn capacity(f: Format) -> f64 {
    recency_span() * (2.0f64).powi(f.mantissa as i32 - 1)
}

/// How many addresses remain if each is to be rewritten `d` times.
pub fn addresses_for(f: Format, rewrites: u64) -> u64 {
    (capacity(f) / rewrites.max(1) as f64).sqrt() as u64
}

/// **The last position at which a rewrite still outranks the write before it**,
/// under the shipped `inv_log_pos`.  `Some(0)` means only a rewrite at the
/// very first position resolves; `None` means not even that one does.
pub fn last_resolving_position(f: Format, addr: u64) -> Option<u64> {
    let a = addr as f64;
    let u = ulp_at(a * a, f.mantissa);
    let resolves = |p: u64| RECENCY_ALPHA * (inv_log_pos(p + 1) - inv_log_pos(p)) > u;
    if !resolves(0) {
        return None;
    }
    let (mut lo, mut hi) = (0u64, 1u64 << 40);
    while lo + 1 < hi {
        let mid = lo + (hi - lo) / 2;
        if resolves(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(lo)
}

/// And the same for a recency term that does not saturate: `span * p / p_max`,
/// whose increments are all `span / p_max`.  It resolves every position up to
/// [`levels`], which is the budget -- the logarithm is what costs the rest.
pub fn linear_capacity(f: Format, addr: u64) -> u64 {
    levels(f, addr) as u64
}

/// The score of a lifted key at a parabolic query, in float32, exactly as
/// `_to_2d_key` and `_to_2d_query` build them.
#[inline]
pub fn score_f32(addr: i64, pos: u64, q: i64) -> f32 {
    let k = addr as f32;
    let kx = 2.0f32 * k;
    let ky = -k * k + (RECENCY_ALPHA as f32) * (inv_log_pos(pos) as f32);
    kx * (q as f32) + ky
}

/// The same in float64.
#[inline]
pub fn score_f64(addr: i64, pos: u64, q: i64) -> f64 {
    let k = addr as f64;
    2.0 * k * (q as f64) - k * k + RECENCY_ALPHA * inv_log_pos(pos)
}

/// Which `(address, position)` a query retrieves, in one format or the other.
pub fn winner(cells: &[(i64, u64)], q: i64, f: Format) -> Option<(i64, u64)> {
    let score = |c: &(i64, u64)| -> f64 {
        if f.mantissa <= 24 {
            score_f32(c.0, c.1, q) as f64
        } else {
            score_f64(c.0, c.1, q)
        }
    };
    cells
        .iter()
        .copied()
        .fold(None, |best: Option<(i64, u64)>, c| match best {
            Some(b) if score(&b) >= score(&c) => Some(b),
            _ => Some(c),
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ceiling::{F32, F64};

    #[test]
    fn the_span_stays_under_the_unit_gap() {
        assert!(recency_span() < 1.0);
        assert!((recency_span() - 0.4328).abs() < 1e-4);
    }

    #[test]
    fn ulp_at_agrees_with_the_hardware() {
        for x in [4096.0f64, 65536.0, 1.05e6, 1.678e7, 1e10] {
            let got = ulp_at(x, 24);
            let a = x as f32;
            let want = (f32::from_bits(a.to_bits() + 1) - a) as f64;
            assert_eq!(got, want, "x = {x}");
        }
    }

    /// **The budget.**
    #[test]
    fn addresses_squared_times_rewrites() {
        assert!((capacity(F32) - 3.63e6).abs() < 1e4);
        assert!((capacity(F64) - 1.95e15).abs() < 1e13);
        assert_eq!(addresses_for(F32, 1), 1905);
        assert_eq!(addresses_for(F32, 1000), 60);
        assert_eq!(addresses_for(F64, 1_000_000), 44149);
    }

    #[test]
    fn levels_at_an_address() {
        assert!((levels(F32, 1024) - 3.46).abs() < 0.01);
        assert!(
            levels(F32, 4096) < 1.0,
            "float32 cannot rewrite at its own wall"
        );
        assert!(levels(F64, 100_000) > 2.2e5);
    }

    /// **The saturation.**  The budget allows `227 000` rewrites at address
    /// `10^5` in float64; consecutive positions stop resolving at `2552`.
    #[test]
    fn inv_log_pos_saturates() {
        assert_eq!(last_resolving_position(F32, 64), Some(40));
        assert_eq!(last_resolving_position(F32, 256), Some(5));
        // only the first position resolves at 1024, and none at all at the wall
        assert_eq!(last_resolving_position(F32, 1024), Some(0));
        assert_eq!(last_resolving_position(F32, 4096), None);
        assert_eq!(last_resolving_position(F64, 100_000), Some(2552));
        assert_eq!(last_resolving_position(F64, 10_000_000), Some(3));

        let ratio = linear_capacity(F64, 100_000) as f64
            / last_resolving_position(F64, 100_000).unwrap() as f64;
        assert!(
            ratio > 80.0,
            "linear recency should be ~90x better, got {ratio}"
        );
    }

    /// And the head itself: two writes to one address, the neighbours beside
    /// it, and the query on the address.  float64 takes the later write at
    /// every size; float32 is already stale at `256`.
    #[test]
    fn the_later_write_wins_only_in_float64() {
        for addr in [64i64, 256, 1024, 1905, 4096] {
            let cells = [(addr, 10u64), (addr, 11), (addr - 1, 12), (addr + 1, 13)];
            assert_eq!(winner(&cells, addr, F64), Some((addr, 11)), "f64 at {addr}");
            let w32 = winner(&cells, addr, F32).unwrap();
            assert_eq!(w32.0, addr, "f32 lost the address itself at {addr}");
            if addr >= 256 {
                assert_eq!(w32.1, 10, "f32 should be stale at {addr}");
            } else {
                assert_eq!(w32.1, 11, "f32 should still resolve at {addr}");
            }
        }
    }
}

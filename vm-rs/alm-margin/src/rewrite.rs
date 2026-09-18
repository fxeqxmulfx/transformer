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
//! **And `inv_log_pos` saturates, so the shipped term reaches nowhere near
//! it.**  The bound above assumes the whole span is available, which would need
//! the writes spaced to use it.  Consecutive positions are what actually
//! happens, and there the increment is `LATEST_ALPHA / ((p + 2) ln^2(p + 2))`,
//! which decays while `ulp(a^2)` does not.
//!
//! [`Recency`] is the choice this costs: [`Recency::InvLogPos`] is what ships,
//! [`Recency::Linear`] spends the same span in equal steps over a horizon fixed
//! in advance.  Neither is free of the budget -- [`levels`] bounds both -- but
//! the linear term reaches it, and the logarithm stops at a ninetieth of it.
//! [`last_resolving_position`] is the measurement: at address `10^5` in float64
//! the shipped term stops ordering rewrites after position **2552**, and a
//! linear term of the same span orders all **226 916** the budget allows.

use crate::ceiling::Format;

/// `LATEST_ALPHA`, `graph/core.py:170`.
pub const RECENCY_ALPHA: f64 = 0.3;

/// `inv_log_pos`, `evaluator.py:230`.  Monotone, and bounded by `1 / ln 2`.
#[inline]
pub fn inv_log_pos(p: u64) -> f64 {
    1.0 / std::f64::consts::LN_2 - 1.0 / ((p + 2) as f64).ln()
}

/// **Which recency feature the key carries.**  The compiler scales whatever
/// sits in this dimension by `LATEST_ALPHA`, so both variants are functions of
/// the position into `[0, 1 / ln 2)` and the choice does not touch the weights
/// -- only the positional feature fed in at inference.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Recency {
    /// What ships: `1/ln 2 - 1/ln(p + 2)`.  Needs no horizon, and saturates.
    InvLogPos,
    /// `(p / horizon) / ln 2`, held at `1 / ln 2` from `horizon` on.  Equal
    /// steps, so it orders every rewrite up to the horizon and none after it;
    /// the price of not saturating is having to name the horizon in advance.
    Linear { horizon: u64 },
}

impl Recency {
    /// The value of the feature dimension at this position.
    #[inline]
    pub fn feature(self, p: u64) -> f64 {
        match self {
            Recency::InvLogPos => inv_log_pos(p),
            Recency::Linear { horizon } => {
                let h = horizon.max(1) as f64;
                (p as f64 / h).min(1.0) / std::f64::consts::LN_2
            }
        }
    }

    /// What the feature contributes to `ky`, after the compiler's scaling.
    #[inline]
    pub fn term(self, p: u64) -> f64 {
        RECENCY_ALPHA * self.feature(p)
    }

    /// How much a rewrite at `p + 1` outranks the write at `p` by, before the
    /// rounding of `-k^2` is allowed to eat it.
    #[inline]
    pub fn step(self, p: u64) -> f64 {
        self.term(p + 1) - self.term(p)
    }
}

/// The whole range the recency term can move a key through: `0.4328`.
///
/// It has to stay below the unit gap between distinct integer keys, which is
/// what makes `0.3` a safe choice and not a tuned one -- with the drift margin
/// of `crate::drift`, the condition on a query drift `d` is
/// `2|d| + span < 1`, so `|d| < 0.284`.  Both variants of [`Recency`] spend
/// exactly this much.
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
/// the writes were spaced to use the whole recency range.  The budget both
/// variants of [`Recency`] live under; only the linear one attains it.
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

/// **The last position at which a rewrite still outranks the write before it.**
/// `Some(0)` means only a rewrite at the very first position resolves; `None`
/// means not even that one does.
///
/// For [`Recency::Linear`] the steps are equal, so the answer is the horizon
/// or nothing at all -- which is the point: the horizon is a choice, and
/// [`largest_horizon`] says how large a choice the address allows.
pub fn last_resolving_position(r: Recency, f: Format, addr: u64) -> Option<u64> {
    let a = addr as f64;
    let u = ulp_at(a * a, f.mantissa);
    if r.step(0) <= u {
        return None;
    }
    if let Recency::Linear { horizon } = r {
        return Some(horizon.max(1) - 1);
    }
    let (mut lo, mut hi) = (0u64, 1u64 << 40);
    while lo + 1 < hi {
        let mid = lo + (hi - lo) / 2;
        if r.step(mid) > u {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(lo)
}

/// **The longest horizon a linear term may span at this address**, which is
/// [`levels`] and so the budget itself.  The logarithm is what costs the rest.
pub fn largest_horizon(f: Format, addr: u64) -> u64 {
    levels(f, addr) as u64
}

/// The score of a lifted key at a parabolic query, in float32, exactly as
/// `_to_2d_key` and `_to_2d_query` build them.
#[inline]
pub fn score_f32(r: Recency, addr: i64, pos: u64, q: i64) -> f32 {
    let k = addr as f32;
    let kx = 2.0f32 * k;
    let ky = -k * k + (r.term(pos) as f32);
    kx * (q as f32) + ky
}

/// The same in float64.
#[inline]
pub fn score_f64(r: Recency, addr: i64, pos: u64, q: i64) -> f64 {
    let k = addr as f64;
    2.0 * k * (q as f64) - k * k + r.term(pos)
}

/// Which `(address, position)` a query retrieves, in one format or the other.
pub fn winner(r: Recency, cells: &[(i64, u64)], q: i64, f: Format) -> Option<(i64, u64)> {
    let score = |c: &(i64, u64)| -> f64 {
        if f.mantissa <= 24 {
            score_f32(r, c.0, c.1, q) as f64
        } else {
            score_f64(r, c.0, c.1, q)
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

    const SHIPPED: Recency = Recency::InvLogPos;

    #[test]
    fn the_span_stays_under_the_unit_gap() {
        assert!(recency_span() < 1.0);
        assert!((recency_span() - 0.4328).abs() < 1e-4);
    }

    /// Both variants spend the same span, so the drift condition does not know
    /// which one is in the key.
    #[test]
    fn both_terms_span_the_same() {
        let lin = Recency::Linear { horizon: 1000 };
        assert!((lin.term(1000) - recency_span()).abs() < 1e-12);
        assert_eq!(lin.term(5000), lin.term(1000));
        assert_eq!(lin.term(0), 0.0);
        assert!(SHIPPED.term(u32::MAX as u64) < recency_span());
        for p in [0u64, 1, 7, 100] {
            assert!(SHIPPED.step(p) > 0.0 && SHIPPED.step(p) < recency_span());
        }
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

    /// **The saturation.**  The budget allows `226 916` rewrites at address
    /// `10^5` in float64; consecutive positions stop resolving at `2552`.
    #[test]
    fn inv_log_pos_saturates() {
        assert_eq!(last_resolving_position(SHIPPED, F32, 64), Some(40));
        assert_eq!(last_resolving_position(SHIPPED, F32, 256), Some(5));
        // only the first position resolves at 1024, and none at all at the wall
        assert_eq!(last_resolving_position(SHIPPED, F32, 1024), Some(0));
        assert_eq!(last_resolving_position(SHIPPED, F32, 4096), None);
        assert_eq!(last_resolving_position(SHIPPED, F64, 100_000), Some(2552));
        assert_eq!(last_resolving_position(SHIPPED, F64, 10_000_000), Some(3));

        let ratio = largest_horizon(F64, 100_000) as f64
            / last_resolving_position(SHIPPED, F64, 100_000).unwrap() as f64;
        assert!(
            ratio > 80.0,
            "linear recency should be ~90x better, got {ratio}"
        );
    }

    /// **And the linear term reaches the budget, to the position.**  At the
    /// budget's own horizon every step still resolves; one address wider, none
    /// of them does.
    #[test]
    fn linear_recency_attains_the_budget() {
        let addr = 100_000u64;
        let h = largest_horizon(F64, addr);
        assert_eq!(h, 226_916);
        assert_eq!(
            last_resolving_position(Recency::Linear { horizon: h }, F64, addr),
            Some(h - 1)
        );
        assert_eq!(
            last_resolving_position(Recency::Linear { horizon: h + 1 }, F64, addr),
            None,
            "one step past the budget nothing resolves at all"
        );
    }

    /// And the head itself: two writes to one address, the neighbours beside
    /// it, and the query on the address.  float64 takes the later write at
    /// every size; float32 is already stale at `256`.
    #[test]
    fn the_later_write_wins_only_in_float64() {
        for addr in [64i64, 256, 1024, 1905, 4096] {
            let cells = [(addr, 10u64), (addr, 11), (addr - 1, 12), (addr + 1, 13)];
            let w = winner(SHIPPED, &cells, addr, F64);
            assert_eq!(w, Some((addr, 11)), "f64 at {addr}");
            let w32 = winner(SHIPPED, &cells, addr, F32).unwrap();
            assert_eq!(w32.0, addr, "f32 lost the address itself at {addr}");
            if addr >= 256 {
                assert_eq!(w32.1, 10, "f32 should be stale at {addr}");
            } else {
                assert_eq!(w32.1, 11, "f32 should still resolve at {addr}");
            }
        }
    }

    /// **The same head, late in the sequence, is where the two terms part.**
    /// At address `10^5` and position `2 * 10^5` the shipped term has long
    /// saturated and the head returns the stale write; the linear term of the
    /// same span returns the fresh one.
    #[test]
    fn late_rewrites_need_the_linear_term() {
        let addr = 100_000i64;
        let lin = Recency::Linear {
            horizon: largest_horizon(F64, addr as u64),
        };
        for p in [10_000u64, 50_000, 200_000, 226_000] {
            let cells = [(addr, p), (addr, p + 1)];
            assert_eq!(
                winner(SHIPPED, &cells, addr, F64),
                Some((addr, p)),
                "inv_log_pos should be stale at {p}"
            );
            assert_eq!(
                winner(lin, &cells, addr, F64),
                Some((addr, p + 1)),
                "linear should be fresh at {p}"
            );
        }
    }
}

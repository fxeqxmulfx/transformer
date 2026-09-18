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

/// **The largest address an inexact query still reaches**, which is lower than
/// the wall an exact one reaches -- `2^25 = 33 554 431` against
/// `crate::ceiling::score_wall`'s `94 906 266` in float64, a factor of
/// `2^1.5`.
///
/// A query off its integer value by `d` needs `2|d| + span < 1` over the
/// reals, and over floats it needs the rounding of `a^2` to fit in what is
/// left too.  Allowing two ulps for that (measured: `1.12` suffices), the
/// allowance survives while `2 * ulp(a^2) < (1 - span) / 2`, that is while
/// `a^2 < 2^(m - 3)`.  Above this the address space is still addressable, but
/// only by a query that is exactly its integer -- so the top `2^1.5` of the
/// range is closed to anything a network computes approximately.
pub fn drift_wall(f: Format) -> u64 {
    let bound = (1.0 - recency_span()) / 2.0;
    debug_assert!((0.2835..0.2836).contains(&bound));
    (2.0f64).powf((f.mantissa as f64 - 3.0) / 2.0).ceil() as u64 - 1
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

    /// **A recency term cannot change which address is retrieved**, and what
    /// the drift is left with is the mantissa again.
    ///
    /// This is the scale separation the span exists for, and it is quantified
    /// over the term's *values*, not over the two variants: any recency
    /// feature whatever, so long as it stays inside `[0, span)`, leaves the
    /// address alone.  The gap between distinct integer keys is `1`, a drift
    /// `d` eats `2|d|` of it, and `span` has to fit in what is left, which
    /// over the reals is `|d| < (1 - span) / 2 = 0.2836`.
    ///
    /// Over float64 it is a little less, and the measurement is how much: the
    /// worst drift that loses the address sits within `1.12 * ulp(a^2)` of the
    /// bound, scanning in eighths of an ulp, so `2 * ulp(a^2)` is safe with
    /// margin.  That is invisible below address `10^5` and eleven percent of
    /// the budget at `10^7`.  The same mantissa that bounds the address space
    /// (`crate::ceiling`) and the rewrite depth ([`levels`]) bounds the
    /// query's accuracy too: three consumers, one budget.
    ///
    /// The loss is not monotone in the drift -- at address `10^6` the address
    /// is lost at `0.283474` and held again at `0.283482` -- so a bisection on
    /// "does it still hold" finds some crossing and not the first one, and the
    /// scan below is a scan for that reason.  Nor is the allowance
    /// conservative: two ulps the other side of the bound the neighbour takes
    /// the query at every address here.
    #[test]
    fn recency_never_moves_the_address() {
        let span = recency_span();
        let bound = (1.0 - span) / 2.0;
        assert!(
            (0.2835..0.2836).contains(&bound),
            "2|d| + span < 1 leaves {bound}"
        );
        let score = |k: f64, term: f64, q: f64| 2.0 * k * q - k * k + term;
        assert_eq!(drift_wall(F64), 33_554_431);
        assert_eq!(drift_wall(crate::ceiling::F32), 1448);
        for a in [64.0f64, 1024.0, 65536.0, 1e6, 1e7, 3.3e7, drift_wall(F64) as f64] {
            // the address carries the least recency there is, every rival the
            // most: the worst case the span can produce.
            let holds = |d: f64| {
                let q = a + d;
                let here = score(a, 0.0, q);
                [-3.0f64, -2.0, -1.0, 1.0, 2.0, 3.0]
                    .iter()
                    .all(|&off| score(a + off, span, q) < here)
            };
            let u = ulp_at(a * a, F64.mantissa);
            let usable = bound - 2.0 * u;
            for i in 0..=400 {
                for sgn in [-1.0f64, 1.0] {
                    let d = sgn * usable * i as f64 / 400.0;
                    assert!(holds(d), "address {a} lost the address at drift {d}");
                }
            }
            // and two ulps the other side of the bound the neighbour takes it,
            // so the allowance is not conservative either.
            let past = bound + 2.0 * u + 1e-12;
            assert!(
                !holds(past) && !holds(-past),
                "at {a} the bound (1 - span) / 2 is conservative"
            );
        }
        // and the wall is where the allowance runs out, not somewhere near it
        let bound_of = |a: u64| 2.0 * ulp_at((a * a) as f64, F64.mantissa);
        assert!(bound_of(drift_wall(F64)) < bound);
        assert!(bound_of(drift_wall(F64) + 1) > bound);
    }

    /// And it is the span that does it, not the arithmetic: at `alpha = 2.4`
    /// the same term reaches past the unit gap and the neighbour wins.
    #[test]
    fn an_oversized_span_does_move_it() {
        let q = 1000.0f64;
        let score = |k: f64, term: f64| 2.0 * k * q - k * k + term;
        let here = score(1000.0, 0.0);
        assert!(score(1001.0, recency_span()) < here, "0.4328 is not enough");
        assert!(score(1001.0, 1.2) > here, "1.2 is");
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

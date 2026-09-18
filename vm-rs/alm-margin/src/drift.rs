//! The true drift margin, and why it does not mention the key bound.
//!
//! `ALM.DriftMargin.qScore_drift` bounds the movement of one score by
//! `2*K*eps` and `order_survives_drift` pays for two of them against a gap of
//! one, which is where `4*K*eps < 1` comes from.  Bounding the two scores
//! separately is what costs the factor: their common part cancels.
//!
//! Write the score as `sScore q k = 2kq - k^2 = q^2 - (k - q)^2`.  Then for two
//! keys `j` and `k`,
//!
//! ```text
//!     sScore q j - sScore q k  =  (j - k) * (2q - j - k)
//! ```
//!
//! and at a query drifted to `q + d` the same difference is
//! `(j - k) * (2q + 2d - j - k)`.  The factor `(j - k)` is common and non-zero,
//! so the sign is decided by `(2q - j - k) + 2d` alone: the difference is zero
//! exactly at `d = (j + k - 2q) / 2`, whatever `j` and `k` are worth.
//!
//! On integer keys at an integer query `j + k - 2q` is an integer.  It is zero
//! exactly when the two keys sit symmetric about the query, which is
//! `symmetric_tie_broken_by_drift` -- there the drift decides the answer and no
//! margin exists.  Everywhere else it is at least one, so
//!
//! ```text
//!     |d| < 1/2   =>   the winner is unchanged
//! ```
//!
//! and that is tight: `flip_point` returns the drift that flips each pair, and
//! the smallest it ever returns on a non-tied pair is `1/2`.

/// The largest drift that leaves every non-tied comparison between integer
/// keys at an integer query intact.
pub const DRIFT_MARGIN: f64 = 0.5;

/// `sScore q k = 2kq - k^2`, in exact integer arithmetic.
#[inline]
pub fn score(q: i128, k: i128) -> i128 {
    2 * k * q - k * k
}

/// The same at a drifted, real query.  `ALM.DriftMargin.qScore`.
#[inline]
pub fn qscore(q: f64, k: i128) -> f64 {
    2.0 * (k as f64) * q - (k as f64) * (k as f64)
}

/// **The drift at which `j` and `k` swap places**, `|j + k - 2q| / 2`.
///
/// `None` when the two keys are symmetric about the query: the comparison is a
/// tie at `d = 0` and decided by the sign of any drift at all.
#[inline]
pub fn flip_point(j: i128, k: i128, q: i128) -> Option<f64> {
    let s = j + k - 2 * q;
    if s == 0 {
        None
    } else {
        Some((s.unsigned_abs() as f64) / 2.0)
    }
}

/// Who wins at an integer query, by exact arithmetic.
pub fn winner_at(keys: &[i128], q: i128) -> Option<i128> {
    keys.iter().copied().max_by_key(|&k| score(q, k))
}

/// What a key set leaves a query: the winner, the key that takes it away
/// first, and the drift at which that happens.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Margin {
    pub winner: i128,
    pub rival: i128,
    /// `None` when winner and rival tie at the undrifted query.
    pub flip: Option<f64>,
}

/// **The margin a whole key set leaves**, as the nearest flip over every rival.
///
/// The claim this crate measures is that `flip` is never below
/// [`DRIFT_MARGIN`] unless it is `None`, and that it does not fall as the keys
/// grow -- which is what `4*K*eps < 1` predicts it would.
pub fn margin_of(keys: &[i128], q: i128) -> Option<Margin> {
    let winner = winner_at(keys, q)?;
    let mut best: Option<Margin> = None;
    for &k in keys {
        if k == winner {
            continue;
        }
        let flip = flip_point(winner, k, q);
        let better = match (&best, flip) {
            (None, _) => true,
            (Some(b), None) => b.flip.is_some(),
            (Some(b), Some(f)) => b.flip.map(|bf| f < bf).unwrap_or(false),
        };
        if better {
            best = Some(Margin {
                winner,
                rival: k,
                flip,
            });
        }
    }
    best
}

/// What `ALM.DriftMargin.order_survives_drift` would allow instead: the largest
/// `eps` satisfying `4*K*eps < 1`, for `K` the bound on the keys.
pub fn lean_bound(keys: &[i128]) -> f64 {
    let k = keys.iter().map(|k| k.unsigned_abs()).max().unwrap_or(0) as f64;
    if k == 0.0 {
        f64::INFINITY
    } else {
        1.0 / (4.0 * k)
    }
}

/// A xorshift64\* good enough to draw key sets with, and no dependency.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform on `-bound ..= bound`.
    pub fn signed(&mut self, bound: i128) -> i128 {
        (self.next_u64() % (2 * bound as u64 + 1)) as i128 - bound
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The factorisation the whole file rests on.
    #[test]
    fn score_difference_factors() {
        let mut rng = Rng::new(7);
        for _ in 0..10_000 {
            let (j, k, q) = (rng.signed(10_000), rng.signed(10_000), rng.signed(10_000));
            assert_eq!(score(q, j) - score(q, k), (j - k) * (2 * q - j - k));
        }
    }

    /// And that it is where the order actually turns over: the two scores are
    /// equal at the signed flip, and on opposite sides of it they disagree.
    #[test]
    fn flip_point_is_where_it_flips() {
        let mut rng = Rng::new(11);
        let mut checked = 0;
        for _ in 0..20_000 {
            let (j, k, q) = (
                rng.signed(100_000),
                rng.signed(100_000),
                rng.signed(100_000),
            );
            if j == k || flip_point(j, k, q).is_none() {
                continue;
            }
            let d = ((j + k - 2 * q) as f64) / 2.0;
            let diff = |t: f64| qscore((q as f64) + t, j) - qscore((q as f64) + t, k);
            assert_eq!(diff(d), 0.0, "j={j} k={k} q={q}");
            assert!(diff(d - 0.25) * diff(d + 0.25) < 0.0, "j={j} k={k} q={q}");
            checked += 1;
        }
        assert!(checked > 15_000);
    }

    /// **The margin is `1/2` and `K` is not in it.**  Key bounds from ten to
    /// ten million, and the bound `ALM.DriftMargin` would ask for alongside.
    #[test]
    fn margin_is_half_at_every_scale() {
        for &bound in &[10i128, 1_000, 1_000_000, 10_000_000] {
            let mut rng = Rng::new(bound as u64 ^ 0x9E37);
            let mut worst = f64::INFINITY;
            let mut tested = 0;
            for _ in 0..4_000 {
                let n = 2 + (rng.next_u64() % 40) as usize;
                let keys: Vec<i128> = (0..n).map(|_| rng.signed(bound)).collect();
                let q = rng.signed(bound);
                let Some(m) = margin_of(&keys, q) else {
                    continue;
                };
                let Some(f) = m.flip else { continue };
                worst = worst.min(f);
                tested += 1;
            }
            assert!(
                tested > 3_000,
                "bound {bound}: only {tested} non-tied cases"
            );
            assert!(worst >= DRIFT_MARGIN, "bound {bound}: worst margin {worst}");
        }
    }

    /// And that it is tight: consecutive keys either side of a query reach it.
    #[test]
    fn half_is_attained() {
        assert_eq!(flip_point(3, 4, 3), Some(0.5));
        assert_eq!(flip_point(3, 4, 4), Some(0.5));
        assert_eq!(flip_point(2, 4, 3), None);
    }
}

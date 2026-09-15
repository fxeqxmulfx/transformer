//! The integer head against the brute-force head, and against exact integer
//! arithmetic where the brute-force head cannot follow.
//!
//! Two claims, and they are different claims.  Below the wall every score is
//! an exact double, `BruteAttentionHead` is the truth, and `LiftAttentionHead`
//! must reproduce it entry for entry -- on live keys, on cleared ones, and on
//! the mixtures the released model actually builds.  Past the wall the stored
//! points are already rounded and the brute-force head is *not* the truth, so
//! the reference there is arithmetic done in `i128` inside the test itself.
//!
//! `todo3.md` section 4 is what the second claim is about, and section 4b the
//! magnitudes: a head keyed on a 32-bit WebAssembly value is past the wall
//! from its first token.

use alm_hull::{BruteAttentionHead, HardAttentionHead, LiftAttentionHead, TieBreak, CLEAR_MARK};

/// xorshift64*, so the cases are reproducible without a dependency.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// `embed_key`'s recency term, as the compiler computes it.
fn offset(pos: u64) -> f64 {
    0.3 * (1.0 / std::f64::consts::LN_2 - 1.0 / ((pos as f64 + 2.0).ln()))
}

/// A live key: `(2k, -k^2 + d)`.
fn live(k: i64, pos: u64) -> [f64; 2] {
    [2.0 * k as f64, -((k * k) as f64) + offset(pos)]
}

/// And the same key with the clear flag set.
fn cleared(k: i64, pos: u64) -> [f64; 2] {
    let l = live(k, pos);
    [l[0], l[1] - CLEAR_MARK]
}

/// Keys small enough that every score below is an exact double, so any
/// disagreement is about which entry the head reached and not about
/// arithmetic: `k^2 <= 10^6` leaves thirty bits under the offsets, and two
/// offsets a position apart differ by `1.6e-5` at the far end of the range.
const RANGE: i64 = 1000;
const POSITIONS: u64 = 500;

/// How many markers a round may hold, and the reason is not this head:
/// at `|k| <= 1000` a cleared ordinate rounds to `-1e30` whatever `k` was, an
/// ulp there being `2^47`, so three markers are three points on one horizontal
/// line -- the degenerate case `todo3.md` section 3 is about, where the
/// envelope drops the interior point and the brute-force head keeps it.  Two
/// are two, and a line through two of them passes nowhere near a live key.
/// `a_run_of_markers_ties_in_the_reference_and_not_in_either_hull` is that
/// case, stated rather than avoided.
const MARKERS: u64 = 2;

#[test]
fn on_the_grid_the_integer_head_answers_what_the_brute_force_head_answers() {
    let mut rng = Rng(0x5eed_1ea7);
    let mut asked = 0;
    for round in 0..8 {
        let mut lift = LiftAttentionHead::new();
        let mut brute = BruteAttentionHead::new();
        let mut markers = 0;
        for seq in 0..80i32 {
            let k = rng.below(RANGE as u64 * 2) as i64 - RANGE;
            let pos = rng.below(POSITIONS);
            let mark = markers < MARKERS && rng.below(8) == 0;
            markers += u64::from(mark);
            let key = if mark { cleared(k, pos) } else { live(k, pos) };
            let val = [rng.below(1000) as f64, rng.below(1000) as f64];
            lift.insert(key, val, seq);
            brute.insert(key, val, seq);
        }
        for _ in 0..120 {
            let qx = rng.below(RANGE as u64 * 4) as f64 - 2.0 * RANGE as f64;
            for qy in [1.0, -1.0, 0.0, 2.0, -0.5] {
                for tb in [TieBreak::Latest, TieBreak::Average] {
                    let q = [qx, qy];
                    assert_eq!(
                        lift.query(q, tb),
                        brute.query(q, tb),
                        "round {round} at {q:?} {tb:?}"
                    );
                    asked += 1;
                }
            }
        }
    }
    assert!(asked > 9_000, "{asked} comparisons");
    // The mixture was real: both families arrived, and neither retired the
    // integer path.
    let mut lift = LiftAttentionHead::new();
    lift.insert(live(3, 4), [1.0, 1.0], 0);
    lift.insert(cleared(3, 5), [2.0, 2.0], 1);
    assert!(lift.on_the_integers());
    assert_eq!(lift.census().keys, 1);
    assert_eq!(lift.census().cleared, 1);
}

/// The exact argmax, in arithmetic no head has: `2qk - k^2` in `i128`, ties
/// broken by the offset, which rises with the position.
fn oracle(keys: &[(i64, u64)], q: i64) -> usize {
    let score = |k: i64| -> i128 { 2 * (q as i128) * (k as i128) - (k as i128) * (k as i128) };
    let mut best = 0;
    for i in 1..keys.len() {
        let (s, b) = (score(keys[i].0), score(keys[best].0));
        if s > b || (s == b && offset(keys[i].1) > offset(keys[best].1)) {
            best = i;
        }
    }
    best
}

#[test]
fn past_the_wall_the_integer_head_is_the_one_that_is_right() {
    // Section 4b's magnitude: the parabolic embedding of a 32-bit value, where
    // `k^2` is `1.1e17` and the stored ordinate has already lost the unit the
    // comparison runs on.
    const V: i64 = 336_860_161;
    let keys: Vec<(i64, u64)> =
        (V - 3..=V + 3).enumerate().map(|(i, k)| (k, 7 * i as u64)).collect();

    let mut lift = LiftAttentionHead::new();
    let mut brute = BruteAttentionHead::new();
    for (seq, &(k, pos)) in keys.iter().enumerate() {
        // The payload names the key, so an answer names the winner.
        let val = [k as f64, pos as f64];
        lift.insert(live(k, pos), val, seq as i32);
        brute.insert(live(k, pos), val, seq as i32);
    }

    let mut brute_wrong = 0;
    for q in V - 2..=V + 2 {
        let want = keys[oracle(&keys, q)];
        let got = lift.query([q as f64, 1.0], TieBreak::Latest);
        assert_eq!(got, Some([want.0 as f64, want.1 as f64]), "the integer head at q = {q}");
        if brute.query([q as f64, 1.0], TieBreak::Latest) != got {
            brute_wrong += 1;
        }
    }
    assert_eq!(brute_wrong, 4, "and four of the five are past what the stored points separate");
    assert_eq!(lift.census().integer, 5, "and every one of them on the integers");
}

#[test]
fn past_the_wall_the_marker_still_decides_and_costs_nothing() {
    // The same magnitude with a clear marker on half the keys.  The live
    // answer has to be exactly the answer the same head gives without them --
    // `ALM.ClearKey.marked_sup'_eq_live` -- and the integer path has to still
    // be the one answering, which is what retiring on the first marker used to
    // cost and what this test exists to hold.
    const V: i64 = 336_860_161;
    let live_keys: Vec<(i64, u64)> =
        (V - 3..=V + 3).step_by(2).enumerate().map(|(i, k)| (k, 7 * i as u64)).collect();

    let (mut with, mut without) = (LiftAttentionHead::new(), LiftAttentionHead::new());
    let mut seq = 0;
    for &(k, pos) in &live_keys {
        let val = [k as f64, pos as f64];
        with.insert(live(k, pos), val, seq);
        without.insert(live(k, pos), val, seq);
        seq += 1;
        // A cleared entry between every pair of live ones, at a key that would
        // win outright were the marker not on it.
        with.insert(cleared(k + 1, pos), [-1.0, -1.0], seq);
        seq += 1;
    }
    assert!(with.on_the_integers(), "the markers did not retire it");
    assert_eq!(with.census().cleared, live_keys.len());

    for q in V - 3..=V + 3 {
        let qq = [q as f64, 1.0];
        let want = live_keys[oracle(&live_keys, q)];
        assert_eq!(
            with.query(qq, TieBreak::Latest),
            Some([want.0 as f64, want.1 as f64]),
            "at q = {q}"
        );
        assert_eq!(with.query(qq, TieBreak::Latest), without.query(qq, TieBreak::Latest));
    }
    assert_eq!(with.census().dominated, 14, "and none of it looked at a marker");
    assert_eq!(with.census().mixed, 0);
}

#[test]
fn a_run_of_markers_ties_in_the_reference_and_not_in_either_hull() {
    // The limit of the test above, named.  Three cleared keys at `|k| <= 1000`
    // are three points on the line `y = -1e30`: the marker costs `2^47` of
    // ordinate (`ALM.ClearKey.the_marker_costs_the_grid`) and `k^2` is far
    // below that, so all three arrive with the same ordinate and score the
    // same at any query with `qx == 0`, and differ only in `kx` otherwise.
    // The brute-force head keeps all three and its tie-break sees them; both
    // hull heads drop the interior one, which is `todo3.md` section 3 and is
    // not something holding the markers aside changes either way.
    let (mut lift, mut brute, mut hull) = (
        LiftAttentionHead::new(),
        BruteAttentionHead::new(),
        HardAttentionHead::new(),
    );
    for (seq, k) in [-3i64, 0, 4].into_iter().enumerate() {
        let val = [k as f64, 0.0];
        lift.insert(cleared(k, seq as u64), val, seq as i32);
        brute.insert(cleared(k, seq as u64), val, seq as i32);
        hull.insert(cleared(k, seq as u64), val, seq as i32);
    }
    // At `qx = 0` every marker scores the same and the reference says so.
    let q = [0.0, -1.0];
    assert_eq!(brute.query(q, TieBreak::Average), Some([1.0 / 3.0, 0.0]), "all three, averaged");
    assert_eq!(lift.query(q, TieBreak::Average), hull.query(q, TieBreak::Average));
    assert_ne!(lift.query(q, TieBreak::Average), brute.query(q, TieBreak::Average));
}

#[test]
fn a_head_of_nothing_but_markers_answers_from_them() {
    // `hL` of `marked_sup'_eq_live` is a hypothesis and not a formality: with
    // no live entry there is nothing to dominate the cleared ones, and the
    // head has to score them.
    let mut lift = LiftAttentionHead::new();
    let mut brute = BruteAttentionHead::new();
    for (seq, k) in [-1i64, 1].into_iter().enumerate() {
        let val = [k as f64, 0.0];
        lift.insert(cleared(k, seq as u64), val, seq as i32);
        brute.insert(cleared(k, seq as u64), val, seq as i32);
    }
    for qi in -8..=8 {
        for qy in [1.0, -1.0, 0.0] {
            let q = [qi as f64, qy];
            assert_eq!(
                lift.query(q, TieBreak::Latest),
                brute.query(q, TieBreak::Latest),
                "at {q:?}"
            );
        }
    }
    assert_eq!(lift.census().dominated, 0, "there was never a live entry to dominate with");
}

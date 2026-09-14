//! The hull head against the brute-force head, on the keys the machine uses.
//!
//! This is the test the C++ release fails: with parabolic keys at a magnitude
//! where the score leaves the integer grid, `HardAttentionHead` and
//! `BruteAttentionHead` disagree (`todo3.md` section 3).  Here the same two
//! heads are compared, and the point of the comparison is that on the grid the
//! disagreement count must be zero at every magnitude.

use alm_hull::{BruteAttentionHead, HardAttentionHead, TieBreak};

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

/// The parabolic embedding of the construction: key `k` goes to `(2k, -k^2)`,
/// query `q` goes to `(q, 1)`, and the score `2qk - k^2` peaks at `k = q`.
fn parabolic_key(k: i64) -> [f64; 2] {
    [2.0 * k as f64, -((k * k) as f64)]
}

fn run_parabolic(seed: u64, range: i64, n_keys: usize, n_queries: usize, tb: TieBreak) -> usize {
    let mut rng = Rng(seed);
    let mut hull = HardAttentionHead::new();
    let mut brute = BruteAttentionHead::new();

    let mut keys = Vec::with_capacity(n_keys);
    for seq in 0..n_keys {
        let k = rng.below(range as u64 * 2) as i64 - range;
        let val = [rng.below(1000) as f64, rng.below(1000) as f64];
        hull.insert(parabolic_key(k), val, seq as i32);
        brute.insert(parabolic_key(k), val, seq as i32);
        keys.push(k);
    }

    let mut disagreements = 0;
    for _ in 0..n_queries {
        // Half the queries land on a stored key, half anywhere in range.
        let q = if rng.next() & 1 == 0 {
            keys[rng.below(keys.len() as u64) as usize]
        } else {
            rng.below(range as u64 * 2) as i64 - range
        };
        let query = [q as f64, 1.0];
        if hull.query(query, tb) != brute.query(query, tb) {
            disagreements += 1;
        }
    }
    disagreements
}

#[test]
fn parabolic_keys_on_the_grid_never_disagree() {
    for (i, range) in [10i64, 1_000, 100_000, 10_000_000].into_iter().enumerate() {
        for tb in [TieBreak::Average, TieBreak::Latest] {
            let d = run_parabolic(0x9E37_79B9_7F4A_7C15 ^ i as u64, range, 400, 2000, tb);
            assert_eq!(d, 0, "range {range}, tie break {tb:?}");
        }
    }
}

/// The exact cross product `(b - a) x (c - a)`, for integer-valued keys.
fn collinear(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> bool {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) == 0.0
}

fn has_three_collinear(keys: &[[f64; 2]]) -> bool {
    for i in 0..keys.len() {
        for j in i + 1..keys.len() {
            for k in j + 1..keys.len() {
                let distinct = keys[i] != keys[j] && keys[j] != keys[k] && keys[i] != keys[k];
                if distinct && collinear(keys[i], keys[j], keys[k]) {
                    return true;
                }
            }
        }
    }
    false
}

#[test]
fn arbitrary_keys_agree_when_no_three_are_collinear() {
    let mut rng = Rng(0xDEAD_BEEF_CAFE_F00D);
    let mut tested = 0;
    for _ in 0..4000 {
        let n = 1 + rng.below(5) as usize;
        let keys: Vec<[f64; 2]> = (0..n)
            .map(|_| [rng.below(11) as f64 - 5.0, rng.below(11) as f64 - 5.0])
            .collect();
        if has_three_collinear(&keys) {
            continue;
        }
        tested += 1;
        let mut hull = HardAttentionHead::new();
        let mut brute = BruteAttentionHead::new();
        for (seq, key) in keys.iter().enumerate() {
            let val = [rng.below(97) as f64, rng.below(97) as f64];
            hull.insert(*key, val, seq as i32);
            brute.insert(*key, val, seq as i32);
        }
        for _ in 0..60 {
            let q = [rng.below(13) as f64 - 6.0, rng.below(13) as f64 - 6.0];
            for tb in [TieBreak::Average, TieBreak::Latest] {
                assert_eq!(
                    hull.query(q, tb),
                    brute.query(q, tb),
                    "keys {keys:?} query {q:?} tie break {tb:?}"
                );
            }
        }
    }
    assert!(tested > 100, "only {tested} configurations were in general position");
}

#[test]
fn three_collinear_keys_are_where_the_hull_head_stops_being_exact() {
    // The hull keeps vertices, not points.  A key that is a maximiser at
    // exactly one query, and interior to the hull everywhere else, is erased
    // on insertion and its payload goes with it.  Every tie in the machine
    // proper is between two keys straddling the query on the parabola
    // `k -> (2k, -k^2)`, which is strictly convex and has no three collinear
    // points; that is the unstated hypothesis under `HardAttentionHead`.
    let keys = [[-5.0, 5.0], [5.0, 5.0], [0.0, 5.0]];
    let mut hull = HardAttentionHead::new();
    let mut brute = BruteAttentionHead::new();
    for (seq, key) in keys.iter().enumerate() {
        hull.insert(*key, [100.0 + seq as f64, 0.0], seq as i32);
        brute.insert(*key, [100.0 + seq as f64, 0.0], seq as i32);
    }
    let q = [0.0, -1.0]; // maximised by all three keys at once
    assert_eq!(brute.query(q, TieBreak::Latest), Some([102.0, 0.0]));
    assert_eq!(hull.query(q, TieBreak::Latest), Some([101.0, 0.0]));
    assert_eq!(brute.query(q, TieBreak::Average), Some([101.0, 0.0]));
    assert_eq!(hull.query(q, TieBreak::Average), Some([100.5, 0.0]));

    // The same three keys on the parabola are in general position, and there
    // the two heads agree at every query.
    let mut hull = HardAttentionHead::new();
    let mut brute = BruteAttentionHead::new();
    for (seq, k) in [-5i64, 5, 0].iter().enumerate() {
        let key = parabolic_key(*k);
        hull.insert(key, [100.0 + seq as f64, 0.0], seq as i32);
        brute.insert(key, [100.0 + seq as f64, 0.0], seq as i32);
    }
    for q in -8..=8 {
        let query = [q as f64, 1.0];
        for tb in [TieBreak::Average, TieBreak::Latest] {
            assert_eq!(hull.query(query, tb), brute.query(query, tb), "q {q} {tb:?}");
        }
    }
}

#[test]
fn an_empty_head_answers_nothing() {
    let hull = HardAttentionHead::new();
    let brute = BruteAttentionHead::new();
    for q in [[1.0, 1.0], [0.0, 0.0], [-3.0, 0.0], [0.0, -2.0]] {
        assert_eq!(hull.query(q, TieBreak::Average), None);
        assert_eq!(brute.query(q, TieBreak::Average), None);
    }
}

#[test]
fn the_envelope_stays_small_while_the_head_grows() {
    // The hull keeps only the maximal lines, so a million parabolic keys drawn
    // from a thousand distinct values leave at most a thousand lines behind.
    let mut rng = Rng(7);
    let mut hull = HardAttentionHead::new();
    for seq in 0..20_000 {
        let k = rng.below(1000) as i64;
        hull.insert(parabolic_key(k), [k as f64, 0.0], seq);
    }
    assert_eq!(hull.len(), 20_000);
    // Every key is on a downward parabola, so every one of the 1000 distinct
    // keys is on the upper envelope and none of the duplicates are.
    let mut brute = BruteAttentionHead::new();
    let mut rng = Rng(7);
    for seq in 0..20_000 {
        let k = rng.below(1000) as i64;
        brute.insert(parabolic_key(k), [k as f64, 0.0], seq);
    }
    for k in 0..1000i64 {
        let q = [k as f64, 1.0];
        assert_eq!(hull.query(q, TieBreak::Latest), brute.query(q, TieBreak::Latest));
    }
}

/// The wall, on a real head: the witness fires exactly where the answer stops
/// being the true `argmax_k 2qk - k^2`.
///
/// `todo3.md` section 4.  Below the wall the head returns the query's own key;
/// at it, the key one *past* the query scores the same, because that key's
/// stored `-k^2` has rounded up by one and the unit it lost is the whole
/// margin.  `TieBreak::Latest` then hands back the wrong entry, silently.
#[test]
fn the_head_reports_the_wall_and_is_wrong_just_past_it() {
    use alm_hull::TieBreak;

    let probe = |q: i64| -> (Option<[f64; 2]>, bool) {
        let mut head = HardAttentionHead::new();
        for (seq, k) in (q - 1..=q + 1).enumerate() {
            // The value carries the key, so the answer names the winner.
            head.insert(parabolic_key(k), [k as f64, 0.0], seq as i32);
        }
        let out = head.query([q as f64, 1.0], TieBreak::Latest);
        (out, head.grid_witness().is_clean())
    };

    // Comfortably below the wall: the query's own key wins, and the head has
    // seen nothing it cannot separate.
    let (out, clean) = probe(1_000_000);
    assert_eq!(out, Some([1_000_000.0, 0.0]));
    assert!(clean);

    // One below the wall: still right, still clean.
    let (out, clean) = probe(94_906_264);
    assert_eq!(out, Some([94_906_264.0, 0.0]));
    assert!(clean);

    // At the wall: the witness fires, and the answer is the key one past the
    // query rather than the query itself.
    let (out, clean) = probe(94_906_266);
    assert!(!clean, "the head has crossed 2^53 and says so");
    assert_eq!(out, Some([94_906_267.0, 0.0]), "the wrong entry, by one");
}

/// The wall is not only about trace length: a head keyed on a 32-bit WebAssembly
/// value is past it from the first token.
///
/// This key and this query scale are taken verbatim from a `hello` run of the
/// released `model.bin` — `q = [4.763922097235979e18, 14142135623.730951]`,
/// winning key `[673720322.0, -1.1347476806894592e17]`, which is the parabolic
/// embedding of the value `336 860 161`.  `todo3.md` section 4 states the wall
/// as 94 906 266 *tokens*; at this scale it is 73 966 031 in whatever the head
/// is keyed on, and a 32-bit value runs to 4.29e9.
#[test]
fn a_head_keyed_on_a_32_bit_value_is_past_the_wall_from_the_start() {
    use alm_hull::TieBreak;

    /// The hard-attention scale the released model queries at.
    const SCALE: f64 = 14142135623.730951;
    const V: i64 = 336860161;

    let mut head = HardAttentionHead::new();
    for (seq, k) in (V - 3..=V + 3).enumerate() {
        head.insert(parabolic_key(k), [k as f64, 0.0], seq as i32);
    }

    // The query asks for V itself, and V is the true argmax of `2qk - k^2`.
    let out = head.query([SCALE * V as f64, SCALE], TieBreak::Average);
    let got = out.expect("the head is not empty")[0];
    assert_ne!(got, V as f64, "the head does not return the key it was asked for");

    // What it returns is drawn from the keys whose rounded scores tie the
    // maximum — V is not among them, its own score rounds *below* theirs — so
    // the answer is off by a whole key, not by an ulp.
    assert!((got - V as f64).abs() >= 1.0, "returned {got}, asked for {V}");

    assert!(!head.grid_witness().is_clean(), "and the head says so");
}

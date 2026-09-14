//! What the envelope costs when the keys do not arrive in a helpful order.
//!
//! The array envelope in `alm_hull::cht` is `O(n)` per insertion where the
//! `std::multiset` it was ported from is `O(log n)`, and on the reference
//! traces that has not cost anything: the deepest insertion the sudoku trace
//! ever makes moves 3 741 lines out of an envelope of 1 055 416.  That is a
//! measurement, not a guarantee, and the difference is the point of this
//! program.  Nothing in the model forces a key coordinate to arrive in any
//! particular order, so the question is what happens when it does not.
//!
//! A head inserts each key into both halves, the lower one negating the
//! slope, so an order that is cheap for one half is expensive for the other.
//! Under the parabolic lift `k -> (2k, -k^2)` the split is total: every key
//! stays on the upper envelope and only the two extremes stay on the lower,
//! so the whole cost is the upper half, and the whole of *that* is the rank
//! of the arriving slope among the slopes already there.
//!
//! ```text
//! alm-stress                    # every order, at 4096, 16384 and 65536 keys
//! alm-stress --n 262144         # one size
//! alm-stress --order descending
//! ```
//!
//! The moved column is exact rather than sampled: with no line ever dropped,
//! an insertion at index `i` of an envelope of `len` rewrites `len - i`
//! lines, which is what `Cht::add_line` hands to `copy_within`.
//!
//! What it finds, for the same keys under five arrival orders:
//!
//! ```text
//!        order         n            moved       mean       max    seconds
//!    ascending    262144           524285        1.0         2      0.043
//!   descending    262144      34359607296    65535.8    262143     97.617
//!     shuffled    262144      17189215144    32785.8    260814     37.334
//!   outside-in    262144      17180000255    32768.2    131071     28.519
//!   inside-out    262144      17180131326    32768.5    262143     50.875
//! ```
//!
//! Four times the keys is twenty-nine times the work, which is the `O(n^2)`
//! the container's asymptotics promise and the reference traces never
//! collect.  Position order costs one move per key; anything else costs
//! `n / 2`.  The whole of the sudoku trace answers its hull in 26 seconds
//! with a million keys; one head fed a quarter of that many in descending
//! order spends 98.  A tree would not care which order they came in.

use std::time::Instant;

use alm_hull::cht::Cht;
use alm_hull::HullMeta;

/// The key of position `k` under the lift the compiler emits: `(2k, -k^2)`.
///
/// Strictly concave in the slope, so no key is ever dropped from the upper
/// envelope and the container does the most work it can be made to do.
fn lift(k: u64) -> (f64, f64) {
    let k = k as f64;
    (2.0 * k, -(k * k))
}

/// xorshift64*, so the shuffled order is reproducible without a dependency.
fn shuffled(n: u64) -> Vec<u64> {
    let mut ks: Vec<u64> = (0..n).collect();
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    for i in (1..ks.len()).rev() {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let r = state.wrapping_mul(0x2545_f491_4f6c_dd1d);
        ks.swap(i, (r % (i as u64 + 1)) as usize);
    }
    ks
}

/// The arrival orders, each a permutation of the same `n` keys.
///
/// They all build the same envelope; only the path to it differs, which is
/// the whole of what is being measured.
fn order(name: &str, n: u64) -> Option<Vec<u64>> {
    Some(match name {
        // Position order, and what every reference trace approximates.
        "ascending" => (0..n).collect(),
        // The mirror of it, which one of the two halves always sees.
        "descending" => (0..n).rev().collect(),
        "shuffled" => shuffled(n),
        // Alternating ends: every key lands at one extreme of the envelope.
        "outside-in" => {
            let (mut lo, mut hi, mut v) = (0, n - 1, Vec::with_capacity(n as usize));
            while lo < hi {
                v.push(lo);
                v.push(hi);
                lo += 1;
                hi -= 1;
            }
            if lo == hi {
                v.push(lo);
            }
            v
        }
        // The reverse: every key lands in the middle of what is there.
        "inside-out" => {
            let mut v: Vec<u64> = (0..n).collect();
            v.sort_by_key(|k| (*k as i64 - (n as i64) / 2).abs());
            v
        }
        _ => return None,
    })
}

/// One half of a head, as `HullHalf` drives it: the lower negates the slope.
fn add(c: &mut Cht, kx: f64, ky: f64, upper: bool, seq: i32) {
    let meta = HullMeta::of([0.0, 0.0], seq);
    if upper {
        c.add_line(kx, ky, meta)
    } else {
        c.add_line(-kx, -ky, meta)
    }
}

/// Where `m` would go: the number of slopes already below it.
fn rank(c: &Cht, m: f64) -> usize {
    let (mut lo, mut hi) = (0, c.len());
    while lo < hi {
        let mid = (lo + hi) / 2;
        if c.get(mid).m.get() < m {
            lo = mid + 1
        } else {
            hi = mid
        }
    }
    lo
}

struct Run {
    upper: usize,
    lower: usize,
    moved: u128,
    max: usize,
    secs: f64,
}

fn run(ks: &[u64]) -> Run {
    // Timed first, on its own pair, so that counting costs the clock nothing.
    let (mut u, mut l) = (Cht::new(), Cht::new());
    let t = Instant::now();
    for (i, &k) in ks.iter().enumerate() {
        let (kx, ky) = lift(k);
        add(&mut u, kx, ky, true, i as i32);
        add(&mut l, kx, ky, false, i as i32);
    }
    let secs = t.elapsed().as_secs_f64();
    let (upper, lower) = (u.len(), l.len());

    // Counted second: the tail each insertion rewrites, exactly.
    let (mut u, mut l) = (Cht::new(), Cht::new());
    let (mut moved, mut max) = (0u128, 0usize);
    for (i, &k) in ks.iter().enumerate() {
        let (kx, ky) = lift(k);
        for (c, upper) in [(&mut u, true), (&mut l, false)] {
            let m = if upper { kx } else { -kx };
            let shift = c.len() - rank(c, m);
            moved += shift as u128;
            max = max.max(shift);
            add(c, kx, ky, upper, i as i32);
        }
    }
    Run { upper, lower, moved, max, secs }
}

const ORDERS: [&str; 5] = ["ascending", "descending", "shuffled", "outside-in", "inside-out"];

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut sizes: Vec<u64> = Vec::new();
    let mut names: Vec<String> = Vec::new();

    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--n" if i + 1 < argv.len() => {
                match argv[i + 1].parse() {
                    Ok(n) if n > 1 => sizes.push(n),
                    _ => {
                        eprintln!("--n wants a count above 1, not {}", argv[i + 1]);
                        std::process::exit(2)
                    }
                }
                i += 1;
            }
            "--order" if i + 1 < argv.len() => {
                if order(&argv[i + 1], 4).is_none() {
                    eprintln!("no such order: {}; have {}", argv[i + 1], ORDERS.join(", "));
                    std::process::exit(2)
                }
                names.push(argv[i + 1].clone());
                i += 1;
            }
            a => {
                eprintln!("usage: alm-stress [--n COUNT]... [--order NAME]...");
                eprintln!("unknown argument: {a}");
                std::process::exit(2)
            }
        }
        i += 1;
    }
    if sizes.is_empty() {
        sizes = vec![4096, 16384, 65536];
    }
    if names.is_empty() {
        names = ORDERS.iter().map(|s| s.to_string()).collect();
    }

    println!("{:>12} {:>9} {:>8} {:>8} {:>16} {:>10} {:>9} {:>9}", "order", "n", "upper", "lower", "moved", "mean", "max", "seconds");
    for &n in &sizes {
        for name in &names {
            let ks = order(name, n).expect("the order was checked when it was parsed");
            let r = run(&ks);
            let mean = r.moved as f64 / (2.0 * n as f64);
            println!("{:>12} {:>9} {:>8} {:>8} {:>16} {:>10.1} {:>9} {:>9.3}", name, n, r.upper, r.lower, r.moved, mean, r.max, r.secs);
        }
    }
}

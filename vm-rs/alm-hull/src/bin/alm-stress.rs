//! What the envelope costs when the keys do not arrive in a helpful order.
//!
//! Nothing in the model forces a key coordinate to arrive in any particular
//! order -- `cache.rs` reads the keys from a learned projection -- so a
//! container that is only fast on the order the reference traces happen to
//! have is one trace away from stopping.  This program asks for the other
//! orders.
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
//! What it finds, for the same 262 144 keys under five arrival orders:
//!
//! ```text
//!        order         n    upper    lower    u.desc    l.desc   seconds
//!    ascending    262144   262144        2         0         0     0.054
//!   descending    262144   262144        2         0         0     0.066
//!     shuffled    262144   262144        2    262118    262118     0.152
//!   outside-in    262144   262144        2    262142    262142     0.078
//!   inside-out    262144   262144        2         0         0     0.060
//! ```
//!
//! Three-fold between the best order and the worst, and the spread does not
//! widen with the size.  The descent columns say where half of it comes from:
//! three of the five orders hand `lower_bound_slope` a slope outside the span
//! the envelope already covers, which the cached ends answer with no descent
//! at all, and those three are the three fastest.  A tariff that counts a
//! search per key cannot see that difference; it is the difference between
//! `2n` comparisons and `n log n` of them.
//!
//! The other half is not comparisons.  `shuffled` and `outside-in` descend on
//! the same number of keys, to within fifty, and one of them takes twice as
//! long: `outside-in` walks a short prefix of the same path every time and
//! `shuffled` walks the whole tree, so what separates those two lines is the
//! memory the identical comparisons reach for.
//!
//! The vector this port used to carry spent 0.040s on the first of those
//! lines and 102.861s on the second, where the same descending order at a
//! quarter of the keys had cost it 3.296s -- four times the keys, thirty-one
//! times the work.  That is why it is no longer here.

use std::time::Instant;

use alm_hull::envelope::Envelope;
use alm_hull::tree::NIL;
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
        // Alternating ends of the key range, which is not the same as the
        // ends of the envelope: after the first two, every key falls strictly
        // inside the span the envelope already covers, and so descends.
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
        // Sorted by distance from the middle, so it starts there -- and then
        // every key is further out than every key before it, which is to say
        // it is a new extreme too, on whichever side it fell.
        "inside-out" => {
            let mut v: Vec<u64> = (0..n).collect();
            v.sort_by_key(|k| (*k as i64 - (n as i64) / 2).abs());
            v
        }
        _ => return None,
    })
}

/// Whether inserting a line of slope `m` would descend the tree.
///
/// The three cases `lower_bound_slope` answers from the cached ends: an empty
/// envelope, a slope above every slope there, a slope below every slope
/// there.  Anything else walks down from the root.  Counting this is the
/// point of the program: the search is the only part of an insertion whose
/// cost grows with the length, so an order that never descends is charged a
/// constant per key and an order that always descends is charged a logarithm.
fn descends(e: &Envelope, m: f64) -> bool {
    let (lo, hi) = (e.first(), e.last());
    hi != NIL && e.get(hi).m.get() >= m && e.get(lo).m.get() < m
}

/// One half of a head, as `HullHalf` drives it: the lower negates the slope.
fn add(e: &mut Envelope, kx: f64, ky: f64, upper: bool, seq: i32) {
    let meta = HullMeta::of([0.0, 0.0], seq);
    if upper {
        e.add_line(kx, ky, meta)
    } else {
        e.add_line(-kx, -ky, meta)
    }
}

struct Run {
    upper: usize,
    lower: usize,
    descents: (usize, usize),
    secs: f64,
}

/// The descent counts of the two halves, taken on a second pass so that the
/// timed loop is the one the library really runs and nothing else.
///
/// They are reported apart because they are not the same price: the upper
/// envelope holds every key, so a descent into it is a walk of `log n` nodes,
/// while the lower holds two and a descent into it is a walk of one.
fn count_descents(ks: &[u64]) -> (usize, usize) {
    let (mut u, mut l) = (Envelope::new(), Envelope::new());
    let (mut du, mut dl) = (0, 0);
    for (i, &k) in ks.iter().enumerate() {
        let (kx, ky) = lift(k);
        du += usize::from(descends(&u, kx));
        dl += usize::from(descends(&l, -kx));
        add(&mut u, kx, ky, true, i as i32);
        add(&mut l, kx, ky, false, i as i32);
    }
    (du, dl)
}

fn run(ks: &[u64]) -> Run {
    let (mut u, mut l) = (Envelope::new(), Envelope::new());
    let t = Instant::now();
    for (i, &k) in ks.iter().enumerate() {
        let (kx, ky) = lift(k);
        add(&mut u, kx, ky, true, i as i32);
        add(&mut l, kx, ky, false, i as i32);
    }
    let secs = t.elapsed().as_secs_f64();
    Run { upper: u.len(), lower: l.len(), descents: count_descents(ks), secs }
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

    println!(
        "{:>12} {:>9} {:>8} {:>8} {:>9} {:>9} {:>9}",
        "order", "n", "upper", "lower", "u.desc", "l.desc", "seconds"
    );
    for &n in &sizes {
        for name in &names {
            let ks = order(name, n).expect("the order was checked when it was parsed");
            let r = run(&ks);
            println!(
                "{:>12} {:>9} {:>8} {:>8} {:>9} {:>9} {:>9.3}",
                name, n, r.upper, r.lower, r.descents.0, r.descents.1, r.secs
            );
        }
    }
}

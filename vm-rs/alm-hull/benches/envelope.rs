//! The envelope, at the shape the running machine actually gives it.
//!
//! What that shape is was measured rather than assumed.  Instrumenting
//! `add_line` over the `sudoku` trace counts 280 740 656 insertions, of which
//!
//!   * 4.8 % descend the tree at all,
//!   * 12.4 % arrive above every slope present and are appended,
//!   * the remaining 82.8 % arrive below every slope present and are
//!     prepended,
//!
//! at an average envelope length of 29 771 and a maximum of 1 055 415 — one
//! line per token, which is `ALM.HullCover` and `ALM.HullLift` in the run: on
//! the paraboloid no erase rule ever fires, so nothing the machine inserts is
//! ever discarded.  The peak resident set of that run is 1.58 GB, so the trees
//! being walked are an order of magnitude past the last level of cache and
//! every parent hop is a load from memory.
//!
//! Hence the sizes and orders here.  `n = 65536` puts the arena near the
//! measured average and well outside a 16 MB L3; the keys are the parabolic
//! lift `(2k, -k^2)` that `alm-stress` uses and `hull_cache.py` stores, so
//! `descending` is the 82.8 % case, `ascending` the 12.4 % one, and `shuffled`
//! the 4.8 %.
//!
//! Source: `vm-rs/alm-hull/src/bin/alm-stress.rs` for the orders and the
//! query patterns, which this reuses so the two tables can be read together.

use std::hint::black_box;

use alm_hull::envelope::Envelope;
use alm_hull::tree::NIL;
use alm_hull::{Break, HullMeta};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

/// The measured average envelope length, rounded to a power of two.
const N: u64 = 65536;

/// The parabolic lift of a key, exactly as `alm-stress` and `hull_cache.py`
/// take it: the line `y = 2k x - k^2`, tangent to the paraboloid at `k`.
fn lift(k: u64) -> (f64, f64) {
    let k = k as f64;
    (2.0 * k, -(k * k))
}

/// xorshift64*, seeded as `alm-stress` seeds it so the orders agree.
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

/// The three orders the trace is made of, in the proportions given above.
fn order(name: &str, n: u64) -> Vec<u64> {
    match name {
        "ascending" => (0..n).collect(),
        "descending" => (0..n).rev().collect(),
        "shuffled" => shuffled(n),
        _ => unreachable!(),
    }
}

const ORDERS: [&str; 3] = ["descending", "ascending", "shuffled"];

fn build(c: &mut Criterion) {
    let mut g = c.benchmark_group("build");
    g.sample_size(20);
    for name in ORDERS {
        let ks: Vec<(f64, f64)> = order(name, N).into_iter().map(lift).collect();
        g.throughput(Throughput::Elements(N));
        g.bench_with_input(BenchmarkId::from_parameter(name), &ks, |b, ks| {
            b.iter_batched_ref(
                Envelope::new,
                |e| {
                    for (i, &(kx, ky)) in ks.iter().enumerate() {
                        e.add_line(kx, ky, HullMeta::of([0.0, 0.0], i as i32));
                    }
                },
                criterion::BatchSize::LargeInput,
            )
        });
    }
    g.finish();
}

/// An envelope of `N` lifted keys, which holds all of them.
fn filled() -> Envelope {
    let mut e = Envelope::new();
    for k in 0..N {
        let (kx, ky) = lift(k);
        e.add_line(kx, ky, HullMeta::of([0.0, 0.0], k as i32));
    }
    e
}

/// The query patterns of `alm-stress`, at the same sizes.
fn pattern(name: &str, n: u64) -> Vec<f64> {
    match name {
        "sweep" => (0..n).map(|k| k as f64).collect(),
        "shuffled" => shuffled(n).into_iter().map(|k| k as f64).collect(),
        "local" => {
            let mut state = 0x243f_6a88_85a3_08d3u64;
            let mut at = (n / 2) as i64;
            (0..n)
                .map(|_| {
                    state ^= state >> 12;
                    state ^= state << 25;
                    state ^= state >> 27;
                    let r = state.wrapping_mul(0x2545_f491_4f6c_dd1d);
                    at = (at + (r % 17) as i64 - 8).clamp(0, n as i64 - 1);
                    at as f64
                })
                .collect()
        }
        "repeat" => vec![(n / 2) as f64; n as usize],
        _ => unreachable!(),
    }
}

const PATTERNS: [&str; 4] = ["sweep", "shuffled", "local", "repeat"];

/// `argmax`, and the one step either side of it that every head takes to
/// collect the ties — the `prev`/`next` of `HullHalf::query`.
fn query(c: &mut Criterion) {
    let e = filled();
    let mut g = c.benchmark_group("query");
    for name in PATTERNS {
        let xs: Vec<Break> = pattern(name, N).into_iter().map(|x| Break::ratio(x, 1.0)).collect();
        g.throughput(Throughput::Elements(N));
        g.bench_with_input(BenchmarkId::from_parameter(name), &xs, |b, xs| {
            b.iter(|| {
                let mut acc = 0u64;
                for &x in xs {
                    let at = e.argmax(black_box(x)).unwrap();
                    acc += u64::from(e.prev(at) != NIL) + u64::from(e.next(at) != NIL);
                }
                acc
            })
        });
    }
    g.finish();
}

criterion_group!(benches, build, query);
criterion_main!(benches);

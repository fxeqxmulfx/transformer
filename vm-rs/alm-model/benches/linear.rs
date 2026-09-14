//! The two kernels `perf` finds at the top of a `sudoku` run.
//!
//! The shapes are the shipped model's and not round numbers: `model.bin`
//! declares `vocab=915 D=38 layers=7 heads=19 d_ffn=47`, so the four dense
//! projections of a layer are `114x38`, `38x38`, `94x38` and `38x47`, and the
//! output head is `915x38` at 85 % zero.  Benchmarking `256x256` instead would
//! measure a different machine: at `cols = 38` the inner product is far too
//! short to amortize anything, and that is the whole difficulty.
//!
//! Weights and inputs are drawn from a fixed xorshift so that two runs of the
//! benchmark compare the same arithmetic, and are shifted off zero so that no
//! value is denormal and no branch predicts.

use std::hint::black_box;

use alm_model::linear::{Dense, SparseHead};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

/// xorshift64*, the generator `alm-stress` uses, so the two agree.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// A finite `f64` in `[-1, 1)`, never zero and never denormal.
    fn f64(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }
}

fn vec_of(rng: &mut Rng, n: usize) -> Vec<f64> {
    (0..n).map(|_| rng.f64()).collect()
}

/// The four projection shapes of one layer, named as `model.rs` names them.
const SHAPES: [(&str, usize, usize); 4] =
    [("qkv", 114, 38), ("out", 38, 38), ("ff_in", 94, 38), ("ff_out", 38, 47)];

fn dense(c: &mut Criterion) {
    let mut g = c.benchmark_group("dense");
    for (name, rows, cols) in SHAPES {
        let mut rng = Rng(0x243f_6a88_85a3_08d3);
        let w = vec_of(&mut rng, rows * cols);
        let x = vec_of(&mut rng, cols);
        let m = Dense::of(&w, rows, cols);
        let mut y = vec![0.0; rows];
        g.throughput(Throughput::Elements((rows * cols) as u64));
        g.bench_with_input(BenchmarkId::from_parameter(name), &m, |b, m| {
            b.iter(|| m.apply(black_box(&x), black_box(&mut y)))
        });
    }
    g.finish();
}

/// And all four together, which is what one layer of a step actually pays.
fn layer(c: &mut Criterion) {
    let mut rng = Rng(0x243f_6a88_85a3_08d3);
    let mut ms = Vec::new();
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for (_, rows, cols) in SHAPES {
        let w = vec_of(&mut rng, rows * cols);
        ms.push(Dense::of(&w, rows, cols));
        xs.push(vec_of(&mut rng, cols));
        ys.push(vec![0.0; rows]);
    }
    let mut g = c.benchmark_group("layer");
    g.throughput(Throughput::Elements(SHAPES.iter().map(|&(_, r, c)| (r * c) as u64).sum()));
    g.bench_function("proj", |b| {
        b.iter(|| {
            for ((m, x), y) in ms.iter().zip(&xs).zip(ys.iter_mut()) {
                m.apply(black_box(x), black_box(y));
            }
        })
    });
    g.finish();
}

/// The output head: one `915`-wide argmax per generated token, over rows whose
/// lengths vary because the sparsity does.
fn head(c: &mut Criterion) {
    let (rows, cols) = (915usize, 38usize);
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    // 85 % zero, laid out row by row as `SparseHead::of` will read it.
    let w: Vec<f64> =
        (0..rows * cols).map(|_| if rng.next() % 100 < 15 { rng.f64() } else { 0.0 }).collect();
    let nnz = w.iter().filter(|v| **v != 0.0).count();
    let h = SparseHead::of(&w, rows, cols);
    let x = vec_of(&mut rng, cols);
    let mut g = c.benchmark_group("head");
    g.throughput(Throughput::Elements(nnz as u64));
    g.bench_function("argmax", |b| b.iter(|| black_box(h.argmax(black_box(&x)))));
    g.finish();
}

criterion_group!(benches, dense, layer, head);
criterion_main!(benches);

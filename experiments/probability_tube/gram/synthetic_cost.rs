//! Timing only synthetic dyadic features; no run file or checkpoint is opened.
use probability_gram::{Covariance, single_thread};
use std::{hint::black_box, time::Instant};
fn main() {
    single_thread();
    let x: Vec<_> = (0..512 * 128)
        .map(|i| ((i * 17 % 257) as f64 - 128.) / 128.)
        .collect();
    let q: Vec<_> = (0..512).map(|i| (1 + i % 16) as f64 / 32.).collect();
    let r: Vec<_> = (0..128).map(|i| (1 + i % 8) as f64 / 4.).collect();
    let started = Instant::now();
    for _ in 0..2000 {
        let c = Covariance::new(black_box(&x), black_box(&q), 128).unwrap();
        black_box(c.bound(&[1.; 128]).unwrap());
        black_box(c.bound(&r).unwrap());
    }
    let seconds = started.elapsed().as_secs_f64();
    println!(
        "synthetic_batches=2000 rows=512 width=128 metrics=2 seconds={seconds:.6} ms_per_batch={:.6}",
        seconds / 2.
    );
}

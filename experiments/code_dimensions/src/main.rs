//! What each character code of a task needs from one layer, and whether it is a first-order
//! stationary point of the product layer's cost among interpolants of a half split. See
//! `README.md`. Usage: `code-dimensions [p] [splits]`, by default `97 5`.
mod codes;
mod field;
mod spectral;

use codes::{Code, Field, TASKS, Task, logs, table_complex, table_mod};
use field::{is_prime, primes_above, rank_mod};
use num_complex::Complex64 as C;
use rayon::prelude::*;
use spectral::{FirstOrder, Svd, first_order, svd};

/// One code, measured: exact ranks over each prime and the float spectrum.
struct Measured {
    code: Code,
    rank: Vec<usize>,
    dim: Vec<usize>,
    float_rank: usize,
    cost: f64,
}

/// The rank of `X` and of `[X | Xᵀ]` over `F_ℓ`. O(p³) time, O(p²) memory.
fn exact(x: &[u64], p: usize, l: u64) -> (usize, usize) {
    let mut joined = Vec::with_capacity(2 * p * p);
    for a in 0..p {
        joined.extend_from_slice(&x[a * p..(a + 1) * p]);
        joined.extend((0..p).map(|b| x[b * p + a]));
    }
    let mut own = x.to_vec();
    (rank_mod(&mut own, p, p, l), rank_mod(&mut joined, p, 2 * p, l))
}

fn measure(task: &Task, code: Code, p: u64, log: &[u32], fields: &[Field]) -> Measured {
    let n = p as usize;
    let (rank, dim) = fields
        .iter()
        .map(|field| exact(&table_mod(task, code, p, log, field), n, field.l))
        .unzip();
    let spectrum = svd(table_complex(task, code, p, log), n, n);
    let cost = 1.5 * spectrum.sigma[..spectrum.rank].iter().map(|s| s.powf(2.0 / 3.0)).sum::<f64>();
    Measured { code, rank, dim, float_rank: spectrum.rank, cost }
}

/// The numerical rank of `[X | Xᵀ]`, through its conjugate transpose. O(sweeps · p³).
fn float_dim(x: &[C], n: usize) -> usize {
    let mut joined = Vec::with_capacity(2 * n * n);
    for a in 0..n {
        joined.extend((0..n).map(|b| x[a + b * n].conj()));
        joined.extend((0..n).map(|b| x[b + a * n].conj()));
    }
    svd(joined, 2 * n, n).rank
}

/// A half split of the `n²` cells, column-major: the first `⌊n²/2⌋` of a shuffle by SplitMix64.
fn half_split(n: usize, seed: u64) -> Vec<bool> {
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    };
    let mut order: Vec<u32> = (0..(n * n) as u32).collect();
    for end in (1..order.len()).rev() {
        let size = end as u64 + 1;
        let threshold = size.wrapping_neg() % size;
        let pick = loop {
            let x = next();
            if x >= threshold {
                break x % size;
            }
        };
        order.swap(end, pick as usize);
    }
    let mut train = vec![false; n * n];
    for &cell in &order[..n * n / 2] {
        train[cell as usize] = true;
    }
    train
}

struct Row {
    task: &'static str,
    family: &'static str,
    codes: usize,
    rank: (usize, usize),
    dim: (usize, usize),
    float_agrees: bool,
    float_dim: usize,
    cost: (f64, f64),
    best: Code,
    tests: Vec<FirstOrder>,
}

fn bounds(values: impl Iterator<Item = usize> + Clone) -> (usize, usize) {
    (values.clone().min().unwrap_or(0), values.max().unwrap_or(0))
}

fn row(
    task: &'static Task,
    family: &'static str,
    list: &[Code],
    p: u64,
    log: &[u32],
    fields: &[Field],
    splits: u64,
) -> Row {
    let measured: Vec<Measured> =
        list.par_iter().map(|&code| measure(task, code, p, log, fields)).collect();
    let rank = bounds(measured.iter().flat_map(|m| m.rank.iter().copied()));
    let dim = bounds(measured.iter().flat_map(|m| m.dim.iter().copied()));
    let float_agrees = measured.iter().all(|m| m.rank.iter().all(|&r| r == m.float_rank));
    let best = measured
        .iter()
        .min_by(|x, y| x.cost.total_cmp(&y.cost))
        .expect("a family has codes");
    let high = measured.iter().map(|m| m.cost).fold(f64::MIN, f64::max);
    let n = p as usize;
    let table = table_complex(task, best.code, p, log);
    let float = float_dim(&table, n);
    let spectrum: Svd = svd(table, n, n);
    let tests = (0..splits)
        .into_par_iter()
        .map(|seed| first_order(&spectrum, n, &half_split(n, seed)))
        .collect();
    Row {
        task: task.name,
        family,
        codes: list.len(),
        rank,
        dim,
        float_agrees,
        float_dim: float,
        cost: (best.cost, high),
        best: best.code,
        tests,
    }
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn span((low, high): (usize, usize)) -> String {
    if low == high { low.to_string() } else { format!("{low}–{high}") }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let p: u64 = args.next().map_or(97, |s| s.parse().expect("p"));
    let splits: u64 = args.next().map_or(5, |s| s.parse().expect("splits"));
    assert!(is_prime(p) && p > 3, "p must be a prime above 3");
    let log = logs(p);
    let primes = primes_above(1 << 30, p * (p - 1), 2);
    let fields: Vec<Field> = primes.iter().map(|&l| Field::new(l, p)).collect();
    let q = (p - 1) as u32;
    let additive: Vec<Code> = (1..p as u32).map(Code::Additive).collect();
    let multiplicative: Vec<Code> =
        (1..q).filter(|&j| gcd(j, q) == 1).map(Code::Multiplicative).collect();

    println!("p = {p}, reductions modulo {primes:?}, {splits} half splits by SplitMix64\n");
    println!("| task | code | codes | rank | dim(col + row) | float dim | cost (cheapest) | cost (dearest) | cheapest | \\|R\\|/\\|G\\|, seed 0 | other seeds | largest \\|R\\| on train | CG steps |");
    println!("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: |");
    let mut agrees = true;
    for task in TASKS {
        for (family, list) in [("ψ", &additive), ("χ", &multiplicative)] {
            let r = row(task, family, list, p, &log, &fields, splits);
            agrees &= r.float_agrees && r.dim.0 == r.dim.1 && r.float_dim == r.dim.0;
            let others: Vec<String> =
                r.tests.iter().skip(1).map(|t| format!("{:.3}", t.ratio)).collect();
            let on_train = r.tests.iter().map(|t| t.on_train).fold(0.0, f64::max);
            let steps = r.tests.iter().map(|t| t.iterations).max().unwrap_or(0);
            println!(
                "| {} | {} | {} | {} | {} | {} | {:.1} | {:.1} | {:?} | {:.3} | {} | {:.1e} | {} |",
                r.task,
                r.family,
                r.codes,
                span(r.rank),
                span(r.dim),
                r.float_dim,
                r.cost.0,
                r.cost.1,
                r.best,
                r.tests.first().map_or(f64::NAN, |t| t.ratio),
                others.join(" "),
                on_train,
                steps,
            );
        }
    }
    println!("\nexact and float ranks agree on every code: {agrees}");
}

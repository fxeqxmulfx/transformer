//! Which parts of a table determine a function of the class `g(u) + h(v) + uᵀ M v` over `F_p`:
//! `review/flat_spectrum.md`, section *What a part of the table determines*. See `README.md`.

mod class;
mod solve;

use arithmetic::reference::SplitMix64;
use class::{Table, Workspace, design_rank, graph_count, min_support};
use rayon::prelude::*;
use serde_json::Value;
use solve::{Completion, complete};

/// How the parts of one table split by the criterion, and how often the count missed the rank.
#[derive(Clone, Copy, Default)]
struct Tally {
    parts: u64,
    disconnected: u64,
    short: u64,
    determined: u64,
    mismatches: u64,
}

impl Tally {
    fn merge(self, o: Self) -> Self {
        Self {
            parts: self.parts + o.parts,
            disconnected: self.disconnected + o.disconnected,
            short: self.short + o.short,
            determined: self.determined + o.determined,
            mismatches: self.mismatches + o.mismatches,
        }
    }

    /// One part. A part inside the zero set of a nonzero function must not count as determined.
    fn of(t: &Table, part: &[u32], w: &mut Workspace, inside_zeros: bool) -> Self {
        let rank = design_rank(t, part, w);
        let (c, s) = graph_count(t, part, w);
        let determined = c == 1 && s == t.couplings();
        Self {
            parts: 1,
            disconnected: u64::from(c > 1),
            short: u64::from(c == 1 && !determined),
            determined: u64::from(determined),
            mismatches: u64::from(rank != t.lines() - c + s || inside_zeros && determined),
        }
    }
}

fn below(rng: &mut SplitMix64, n: usize) -> usize {
    let n = n as u64;
    let threshold = n.wrapping_neg() % n;
    loop {
        let x = rng.next();
        if x >= threshold {
            return (x % n) as usize;
        }
    }
}

/// `m` distinct cells, uniformly, by a partial Fisher–Yates shuffle of a fresh pool. O(cells).
fn draw_cells(rng: &mut SplitMix64, pool: &mut [u32], m: usize, part: &mut Vec<u32>) {
    for (i, x) in pool.iter_mut().enumerate() {
        *x = i as u32;
    }
    for i in 0..m {
        pool.swap(i, i + below(rng, pool.len() - i));
    }
    part.extend_from_slice(&pool[..m]);
}

/// The zero set of a random function with `M ≠ 0`, or a random three quarters of it.
fn draw_zero_set(rng: &mut SplitMix64, t: &Table, part: &mut Vec<u32>) {
    let p = usize::from(t.p);
    let g: Vec<usize> = (0..t.rows).map(|_| below(rng, p)).collect();
    let h: Vec<usize> = (0..t.columns).map(|_| below(rng, p)).collect();
    let mut m: Vec<usize> = (0..t.couplings()).map(|_| below(rng, p)).collect();
    m[0] = 1 + below(rng, p - 1);
    let thin = rng.next() & 1 == 1;
    for cell in 0..t.cells() {
        let mixed: usize = t.product(cell).iter().zip(&m).map(|(&x, &c)| usize::from(x) * c).sum();
        let value = g[cell / t.columns] + h[cell % t.columns] + mixed;
        if value % p == 0 && (!thin || rng.next() & 3 != 0) {
            part.push(cell as u32);
        }
    }
}

/// Every subset of the cells. O(2^cells) parts.
fn exhaustive(t: &Table) -> Tally {
    let cells = t.cells() as u32;
    assert!(cells <= 26);
    (0u64..1 << cells)
        .into_par_iter()
        .map_init(
            || (Workspace::new(t), Vec::with_capacity(cells as usize)),
            |(w, part), mask| {
                part.clear();
                part.extend((0..cells).filter(|i| mask >> i & 1 == 1));
                Tally::of(t, part, w, false)
            },
        )
        .reduce(Tally::default, Tally::merge)
}

/// Three families of parts in turn: uniform ones of `n − 1` to `3 n` cells, the cross with up to
/// `2 k l` random cells, and parts inside the zero set of a random function with `M ≠ 0`.
fn sampled(t: &Table, draws: u64, seed: u64) -> Tally {
    (0..draws)
        .into_par_iter()
        .map_init(
            || (Workspace::new(t), Vec::new(), vec![0u32; t.cells()]),
            |(w, part, pool), draw| {
                let mut rng = SplitMix64(seed ^ (draw + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                part.clear();
                let n = t.lines();
                match draw % 3 {
                    0 => {
                        let high = t.cells().min(3 * n);
                        let m = n - 1 + below(&mut rng, high - n + 2);
                        draw_cells(&mut rng, pool, m, part);
                    }
                    1 => {
                        part.extend((0..t.rows).map(|u| (u * t.columns) as u32));
                        part.extend(1..t.columns as u32);
                        for _ in 0..below(&mut rng, 2 * t.couplings() + 1) {
                            part.push(below(&mut rng, t.cells()) as u32);
                        }
                    }
                    _ => draw_zero_set(&mut rng, t, part),
                }
                Tally::of(t, part, w, draw % 3 == 2)
            },
        )
        .reduce(Tally::default, Tally::merge)
}

/// Among `draws` uniform parts of `m` cells: how many determine the class at `k = l = 1`, and how
/// many leave no row or column empty.
fn shares(t: &Table, m: usize, draws: u64) -> (u64, u64) {
    (0..draws)
        .into_par_iter()
        .map_init(
            || (Workspace::new(t), Vec::new(), vec![0u32; t.cells()], vec![false; t.lines()]),
            |(w, part, pool, seen), draw| {
                let mut rng =
                    SplitMix64((m as u64) << 32 ^ (draw + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                part.clear();
                draw_cells(&mut rng, pool, m, part);
                seen.fill(false);
                for &cell in part.iter() {
                    let cell = cell as usize;
                    seen[cell / t.columns] = true;
                    seen[t.rows + cell % t.columns] = true;
                }
                let determined = graph_count(t, part, w) == (1, 1);
                (u64::from(determined), u64::from(seen.iter().all(|&x| x)))
            },
        )
        .reduce(|| (0, 0), |x, y| (x.0 + y.0, x.1 + y.1))
}

fn print_tally(label: &str, (p, k, l): (u8, usize, usize), x: Tally) {
    println!(
        "{p:>3} {k} {l}  {label:<10} {:>10} {:>12} {:>9} {:>10} {:>10}",
        x.parts, x.disconnected, x.short, x.determined, x.mismatches
    );
}

fn check_counts() {
    println!("1. The count n − c + s against the rank of the evaluation matrix");
    println!("  p k l  parts of    number  disconnected  s < k l  determined  mismatches");
    for shape in [(2, 1, 1), (2, 2, 1), (2, 1, 2), (3, 1, 1), (2, 2, 2), (5, 1, 1)] {
        print_tally("all", shape, exhaustive(&Table::new(shape.0, shape.1, shape.2)));
    }
    for (shape, draws) in [
        ((7, 1, 1), 60_000),
        ((11, 1, 1), 30_000),
        ((3, 2, 1), 60_000),
        ((3, 1, 2), 60_000),
        ((3, 2, 2), 30_000),
        ((5, 2, 1), 30_000),
        ((2, 3, 2), 30_000),
        ((97, 1, 1), 600),
    ] {
        print_tally("sampled", shape, sampled(&Table::new(shape.0, shape.1, shape.2), draws, 1));
    }
}

fn check_extremes() {
    println!("\n2. The least part: the cross with the cells (e_i, e_j)");
    for (p, k, l) in [(3, 1, 1), (5, 1, 1), (97, 1, 1), (3, 2, 2), (5, 2, 1)] {
        let t = Table::new(p, k, l);
        let part = t.least_part();
        let rank = design_rank(&t, &part, &mut Workspace::new(&t));
        println!("  p={p} k={k} l={l}: {} cells, rank {rank}, dim {}", part.len(), t.dim());
    }

    println!("\n3. The least number of nonzero cells of a nonzero function, over all functions");
    for (p, k, l) in [
        (2, 1, 1),
        (2, 2, 1),
        (2, 1, 2),
        (2, 2, 2),
        (2, 3, 1),
        (2, 3, 2),
        (3, 1, 1),
        (3, 2, 1),
        (3, 1, 2),
        (5, 1, 1),
    ] {
        let t = Table::new(p, k, l);
        let line = t.rows.min(t.columns);
        println!(
            "  p={p} k={k} l={l}: {} functions, least {}, min(p^k, p^l) = {line}",
            u64::from(p).pow(t.dim() as u32),
            min_support(&t)
        );
    }

    let three = Table::new(3, 1, 1);
    let mut w = Workspace::new(&three);
    let six = [0, 3, 5, 8, 7, 1];
    let (c, s) = graph_count(&three, &six, &mut w);
    let rank = design_rank(&three, &six, &mut w);
    println!("\n4. p = 3, the six cells off the line a + b = 2: c = {c}, s = {s}, rank {rank} of 6");

    let t = Table::new(97, 1, 1);
    let mut w = Workspace::new(&t);
    let all_but_row: Vec<u32> = (97..9409).collect();
    let plus_one: Vec<u32> = (96..9409).collect();
    println!("\n5. p = 97, dim {}", t.dim());
    println!(
        "  all but row 0: {} cells, rank {}",
        all_but_row.len(),
        design_rank(&t, &all_but_row, &mut w)
    );
    println!(
        "  plus one cell: {} cells, rank {}",
        plus_one.len(),
        design_rank(&t, &plus_one, &mut w)
    );

    println!("\n6. p = 97, uniform parts of m cells, 5000 draws each");
    println!("      m  determine  no empty line");
    for m in [194, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 4704] {
        let (determined, full) = shares(&t, m, 5000);
        println!("  {m:>5}  {:>9.4}  {:>13.4}", determined as f64 / 5000.0, full as f64 / 5000.0);
    }
}

/// Completes each task of the 54-case arithmetic data from its frozen training half, the split
/// by SplitMix64(1) that E6 trains on, and scores the held-out half.
fn check_frozen_splits() -> arithmetic::Result<()> {
    println!("\n7. The frozen half splits: one linear solve on the training cells");
    let cases: Vec<Value> = serde_json::from_str(arithmetic::CASES)?;
    let (mut available, mut outside, mut undetermined) = (0, 0, 0);
    for row in &cases {
        let Some(d) = arithmetic::dataset::build(row)? else {
            continue;
        };
        available += 1;
        let p = d.training.vocabulary as u32;
        let mut present = vec![false; 2 * p as usize];
        for &[a, b, _] in d.training.rows.iter().chain(&d.validation) {
            present[a as usize] = true;
            present[(p + b) as usize] = true;
        }
        let fit = complete(p, &d.training.rows, &present);
        let whole: Vec<[u32; 3]> =
            d.training.rows.iter().chain(&d.validation).copied().collect();
        let in_class = !matches!(complete(p, &whole, &present), Completion::NoMember);
        let name = row["case"]["name"].as_str().ok_or("name")?;
        match &fit {
            Completion::NoMember => {
                outside += 1;
                assert!(!in_class, "{name}: the whole table fits and its half does not");
            }
            Completion::NotDetermined => {
                undetermined += 1;
                println!("  {name}: fits, not determined (whole table in class: {in_class})");
            }
            Completion::Determined { m, .. } => {
                let hits = d
                    .validation
                    .iter()
                    .filter(|&&[a, b, y]| fit.predict(p, a, b) == Some(y))
                    .count();
                println!(
                    "  {name}: p = {p}, m = {m}, {} training cells, held out {hits} / {} \
                     (whole table in class: {in_class})",
                    d.training.rows.len(),
                    d.validation.len()
                );
            }
        }
    }
    println!(
        "  {available} tasks with data: {} completed, {undetermined} not determined, \
         {outside} with no function of the class on the training cells",
        available - outside - undetermined
    );
    Ok(())
}

fn main() -> arithmetic::Result<()> {
    check_counts();
    check_extremes();
    check_frozen_splits()
}

#[cfg(test)]
mod tests {
    use crate::solve::{Completion, complete};
    use arithmetic::dataset::Dataset;
    use serde_json::Value;

    /// The twelve formulas of `TablePart.E6.tasks` in Lean, in the same order. `49 = 1/2` mod 97.
    const TASKS: [fn(u32, u32) -> u32; 12] = [
        |a, b| a + b,
        |a, b| a + 97 - b,
        |a, b| (a + b) * 49,
        |a, b| a * b,
        |a, b| a + a * b,
        |a, b| a * a + a * b,
        |a, b| a * a + a * b + b * b,
        |a, b| a * a + a * b + b * b + a,
        |a, b| a * a + b * b,
        |a, b| (a + b) * (a + b),
        |a, b| a * a * a + b,
        |a, b| a * a * a + a * b,
    ];

    /// The tasks with the full table at `p = 97`, by name, on their frozen splits.
    fn full_tables() -> arithmetic::Result<Vec<(String, Dataset)>> {
        let cases: Vec<Value> = serde_json::from_str(arithmetic::CASES)?;
        let mut tables = Vec::with_capacity(cases.len());
        for row in &cases {
            if let Some(d) = arithmetic::dataset::build(row)?
                && d.training.vocabulary == 97
                && d.training.rows.len() == 4704
            {
                let name = row["case"]["name"].as_str().ok_or("name")?;
                tables.push((name.to_owned(), d));
            }
        }
        Ok(tables)
    }

    /// The refutations of `E6Witnesses.lean`, read from the Lean source: a task and two rectangles
    /// `(a, a', b, b', y(a, b), y(a', b), y(a', b'), y(a, b'))`.
    fn lean_witnesses() -> Vec<(&'static str, [u32; 16])> {
        include_str!("../../../src/Transformer/Grokking/FlatSpectrum/TablePart/E6Witnesses.lean")
            .lines()
            .filter_map(|line| {
                let (_, rest) = line.split_once("(\"")?;
                let (name, rest) = rest.split_once('"')?;
                let numbers: Vec<u32> = rest
                    .split(|c: char| !c.is_ascii_digit())
                    .filter(|x| !x.is_empty())
                    .map(|x| x.parse().expect("a number"))
                    .collect();
                Some((
                    name,
                    <[u32; 16]>::try_from(numbers).expect("two rectangles"),
                ))
            })
            .collect()
    }

    /// The order of the cells in the split E6 trains on, as FNV-1a over `97 a + b`. The Lean theorem
    /// `TablePart.E6.fingerprint` proves the same value for its port of the shuffle.
    #[test]
    fn lean_port_of_the_e6_shuffle() -> arithmetic::Result<()> {
        let tables = full_tables()?;
        assert_eq!(tables.len(), 33);
        for (_, d) in &tables {
            let hash = d
                .training
                .rows
                .iter()
                .chain(&d.validation)
                .fold(0xcbf2_9ce4_8422_2325_u64, |h, &[a, b, _]| {
                    (h ^ u64::from(97 * a + b)).wrapping_mul(0x0100_0000_01b3)
                });
            assert_eq!(hash, 359_723_728_075_922_973);
        }
        Ok(())
    }

    /// Each formula of `TablePart.E6.tasks` is the table of exactly one task of the data.
    #[test]
    fn lean_tasks_are_tables_of_the_data() -> arithmetic::Result<()> {
        let tables = full_tables()?;
        for f in TASKS {
            let matching = tables
                .iter()
                .filter(|(_, d)| {
                    d.training
                        .rows
                        .iter()
                        .chain(&d.validation)
                        .all(|&[a, b, y]| f(a, b) % 97 == y)
                })
                .count();
            assert_eq!(matching, 1);
        }
        Ok(())
    }

    /// Every cell and label of `E6Witnesses.lean` is a training row of its task, and the witnesses
    /// are exactly the tasks of the full table that the solve puts outside the class.
    #[test]
    fn lean_witnesses_are_training_cells() -> arithmetic::Result<()> {
        let witnesses = lean_witnesses();
        let mut outside = 0;
        for (name, d) in &full_tables()? {
            let mut present = vec![false; 2 * 97];
            for &[a, b, _] in d.training.rows.iter().chain(&d.validation) {
                present[a as usize] = true;
                present[97 + b as usize] = true;
            }
            let none = matches!(
                complete(97, &d.training.rows, &present),
                Completion::NoMember
            );
            let witness = witnesses.iter().find(|w| w.0 == name);
            assert_eq!(none, witness.is_some(), "{name}");
            let Some((_, w)) = witness else {
                continue;
            };
            outside += 1;
            for r in w.chunks_exact(8) {
                for cell in [
                    [r[0], r[2], r[4]],
                    [r[1], r[2], r[5]],
                    [r[1], r[3], r[6]],
                    [r[0], r[3], r[7]],
                ] {
                    assert!(d.training.rows.contains(&cell), "{name}: {cell:?}");
                }
            }
        }
        assert_eq!(outside, witnesses.len());
        Ok(())
    }
}

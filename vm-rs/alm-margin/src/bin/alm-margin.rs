//! Prints the two measurements the crate exists for: what the drift margin
//! really is, and where the score stops fitting the format.

use alm_margin::ceiling::{first_failing_key_f32, score_wall, F32, F64, FORMATS};
use alm_margin::drift::{lean_bound, margin_of, Rng};
use alm_margin::rewrite::{
    addresses_for, capacity, last_resolving_position, levels, linear_capacity,
};
use alm_margin::DRIFT_MARGIN;

fn main() {
    println!("drift margin, against ALM.DriftMargin's 4*K*eps < 1\n");
    println!(
        "{:>12} {:>8} {:>12} {:>12} {:>10}",
        "key bound", "cases", "measured", "4*K*eps<1", "ratio"
    );
    for &bound in &[10i128, 1_000, 100_000, 10_000_000] {
        let mut rng = Rng::new(bound as u64 ^ 0x9E37);
        let (mut worst, mut tested, mut ties) = (f64::INFINITY, 0u32, 0u32);
        let mut lean = f64::INFINITY;
        for _ in 0..20_000 {
            let n = 2 + (rng.next_u64() % 40) as usize;
            let keys: Vec<i128> = (0..n).map(|_| rng.signed(bound)).collect();
            let q = rng.signed(bound);
            lean = lean.min(lean_bound(&keys));
            match margin_of(&keys, q).and_then(|m| m.flip) {
                Some(f) => {
                    worst = worst.min(f);
                    tested += 1;
                }
                None => ties += 1,
            }
        }
        println!(
            "{:>12} {:>8} {:>12.3} {:>12.2e} {:>9.1e}x",
            bound,
            tested,
            worst,
            lean,
            worst / lean
        );
        if ties > 0 {
            println!("{:>12} {:>8} symmetric ties, no margin exists", "", ties);
        }
    }
    println!("\n  the measured margin is {DRIFT_MARGIN} at every scale: the key bound is");
    println!("  not in it.  a float32 rounding of an address of 10^5 is 8e-3,");
    println!("  four orders inside it.  a run whose worst case reads above 0.5 only");
    println!("  drew no two keys close enough to reach it.\n");

    println!("score wall: the largest key bound whose scores the format separates\n");
    println!(
        "{:>10} {:>10} {:>14} {:>12}",
        "format", "mantissa", "wall", "measured"
    );
    for f in FORMATS {
        let m = if f.mantissa == 24 {
            format!("{}", first_failing_key_f32())
        } else {
            "-".to_string()
        };
        println!(
            "{:>10} {:>10} {:>14} {:>12}",
            f.name,
            f.mantissa,
            score_wall(f),
            m
        );
    }
    println!("\n  float32 holds 4096 addresses, not 2^24: the score squares the key.");
    println!("  this is what forces KEY_OFFSET, and it is not a drift problem.");

    println!("\n\nrewriting an address: the recency term shares that same mantissa\n");
    println!(
        "{:>10} {:>12} {:>12} {:>14} {:>14}",
        "address", "levels f32", "levels f64", "last pos f32", "last pos f64"
    );
    let show = |o: Option<u64>| match o {
        Some(p) => p.to_string(),
        None => "none".to_string(),
    };
    for a in [64u64, 256, 1024, 4096, 100_000, 10_000_000] {
        println!(
            "{:>10} {:>12.1} {:>12.3e} {:>14} {:>14}",
            a,
            levels(F32, a),
            levels(F64, a),
            show(last_resolving_position(F32, a)),
            show(last_resolving_position(F64, a)),
        );
    }
    println!(
        "\n  levels is the budget: addresses^2 * rewrites <= {:.3e} (f32), {:.3e} (f64).",
        capacity(F32),
        capacity(F64)
    );
    for (name, f) in [("float32", F32), ("float64", F64)] {
        let row: Vec<String> = [1u64, 10, 1000, 1_000_000]
            .iter()
            .map(|&d| format!("{} x{}", addresses_for(f, d), d))
            .collect();
        println!("    {name}: {}", row.join(",  "));
    }
    println!("\n  last pos is what the shipped inv_log_pos actually delivers: it");
    println!("  saturates, so consecutive rewrites stop resolving long before the");
    println!("  budget runs out.  a linear recency term of the same span would");
    println!(
        "  reach it -- {} rewrites at address 10^5 in float64, not {}.",
        linear_capacity(F64, 100_000),
        show(last_resolving_position(F64, 100_000))
    );
}

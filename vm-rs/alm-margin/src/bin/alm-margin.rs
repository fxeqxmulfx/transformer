//! Prints the two measurements the crate exists for: what the drift margin
//! really is, and where the score stops fitting the format.

use alm_margin::ceiling::{first_failing_key_f32, score_wall, FORMATS};
use alm_margin::drift::{lean_bound, margin_of, Rng};
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
}

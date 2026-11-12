//! Synthetic numerical timing only: no stress labels or held-out results.
use convex_stress_moments::{Result, projection::Projector};
use std::time::Instant;
fn main() -> Result<()> {
    let n = 97;
    let mut projector = Projector::new(n, 8)?;
    let mut x: Vec<_> = (0..n * n * n)
        .map(|i| ((i * 17 + 3) as f64).sin())
        .collect();
    let start = Instant::now();
    for _ in 0..10 {
        x = projector.ball(&x, (n as f64).sqrt())?;
    }
    println!(
        "10 synthetic n=97 moment projections: {:.6} seconds",
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

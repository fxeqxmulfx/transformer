//! Replay frozen cautious training and inspect its final frequency criterion.
use convex_rotary_recurrence::{Result, diagnostic};
use std::path::Path;
fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 5 {
        return Err("usage: diagnose WORKSPACE STORAGE COHORT OUTPUT mini|forward".into());
    }
    diagnostic::run(
        Path::new(&args[0]),
        Path::new(&args[1]),
        Path::new(&args[2]),
        Path::new(&args[3]),
        &args[4],
    )
}
fn main() {
    if let Err(e) = run() {
        eprintln!("REFUSED: {e}");
        std::process::exit(1);
    }
}

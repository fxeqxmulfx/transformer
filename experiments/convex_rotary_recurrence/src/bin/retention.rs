use convex_rotary_recurrence::{Result, retention};
use std::path::Path;
fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 6 {
        return Err(
            "usage: retention WORKSPACE STORAGE COHORT FRESH_OUTPUT cautious|refined mini|forward"
                .into(),
        );
    }
    retention::run(
        Path::new(&args[0]),
        Path::new(&args[1]),
        Path::new(&args[2]),
        Path::new(&args[3]),
        &args[4],
        &args[5],
    )
}
fn main() {
    if let Err(e) = run() {
        eprintln!("REFUSED: {e}");
        std::process::exit(1);
    }
}

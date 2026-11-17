use convex_recurrent_readouts::{Result, experiment, score};
use std::path::Path;
fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() == 1 && args[0] == "plan" {
        println!(
            "{}",
            serde_json::json!({"protocol":serde_json::from_str::<serde_json::Value>(experiment::PROTOCOL)?,"source_fingerprint":experiment::fingerprint()})
        );
        return Ok(());
    }
    if args.len() != 7 {
        return Err("usage: stress plan | stress train|score|verify WORKSPACE STORAGE SOURCE_RUNS COHORT cautious|refined mini|forward".into());
    }
    match args[0].as_str() {
        "train" => experiment::train(
            Path::new(&args[1]),
            Path::new(&args[2]),
            Path::new(&args[3]),
            Path::new(&args[4]),
            &args[5],
            &args[6],
        ),
        "score" | "verify" => score::run(
            Path::new(&args[1]),
            Path::new(&args[2]),
            Path::new(&args[4]),
            &args[5],
            &args[6],
            args[0] == "verify",
        ),
        _ => Err("unknown frozen refit command".into()),
    }
}
fn main() {
    if let Err(e) = run() {
        eprintln!("REFUSED: {e}");
        std::process::exit(1);
    }
}

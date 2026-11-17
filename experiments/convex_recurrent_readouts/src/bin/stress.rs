use convex_recurrent_readouts::{Result, experiment, score};
use std::path::Path;
fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let raw = args.first().map_or("", String::as_str);
    let (command, text) = if let Some(command) = raw.strip_suffix("-projected") {
        (command, experiment::PROJECTED_PROTOCOL)
    } else if let Some(command) = raw.strip_suffix("-long") {
        (command, experiment::LONG_PROTOCOL)
    } else {
        (raw, experiment::PROTOCOL)
    };
    if args.len() == 1 && command == "plan" {
        println!(
            "{}",
            serde_json::json!({"protocol":serde_json::from_str::<serde_json::Value>(text)?,"source_fingerprint":experiment::fingerprint()})
        );
        return Ok(());
    }
    if args.len() != 7 {
        return Err("usage: stress plan[-long|-projected] | stress train|score|verify[-long|-projected] WORKSPACE STORAGE SOURCE_RUNS COHORT cautious|refined mini|forward".into());
    }
    match command {
        "train" => experiment::train(
            Path::new(&args[1]),
            Path::new(&args[2]),
            Path::new(&args[3]),
            Path::new(&args[4]),
            &args[5],
            &args[6],
            text,
        ),
        "score" | "verify" => score::run(
            Path::new(&args[1]),
            Path::new(&args[2]),
            Path::new(&args[4]),
            &args[5],
            &args[6],
            command == "verify",
            text,
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

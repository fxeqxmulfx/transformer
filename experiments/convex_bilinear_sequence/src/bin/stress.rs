//! Native training/scoring entry point. No task identity enters model inference.
use convex_bilinear_sequence::{
    Result,
    experiment::{self, score, train_run},
};
use serde_json::{Value, json};
use std::path::Path;

fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let command = args.first().map(String::as_str).unwrap_or("");
    let protocol_text = if command.ends_with("-scaled") {
        experiment::SCALED_PROTOCOL
    } else {
        experiment::PROTOCOL
    };
    if matches!(command, "plan" | "plan-scaled") && args.len() == 1 {
        let protocol: Value = serde_json::from_str(protocol_text)?;
        for case in protocol["cases"].as_array().ok_or("cases")? {
            let name = case["name"].as_str().ok_or("name")?;
            let (_, original, task) = experiment::case_in(name, protocol_text)?;
            let shape = experiment::shape(&task)?;
            let mut support = std::collections::BTreeSet::new();
            for example in &task.splits["train"] {
                for (position, &label) in example.targets.iter().enumerate() {
                    if label >= 0 {
                        support.extend(
                            shape
                                .addresses(&example.tokens[..=position])?
                                .columns
                                .into_iter()
                                .map(|(d, _)| d),
                        );
                    }
                }
            }
            println!(
                "{}",
                json!({"case":name,"shape":shape,"complete_real_coordinates":shape.real_coordinates().to_string(),
                "training_support_columns":support.len(),"reservation_bytes":shape.reservation(support.len())?,
                "batch":original["case"]["batch"],"training_examples":task.splits["train"].len(),
                "selection":task.selection,"source_fingerprint":experiment::fingerprint()})
            );
        }
        return Ok(());
    }
    if args.len() != 7 {
        return Err("usage: stress plan[-scaled] | stress train|score|verify[-scaled] WORKSPACE STORAGE COHORT CASE mini|forward SEED".into());
    }
    let workspace = Path::new(&args[1]);
    let storage = Path::new(&args[2]);
    let cohort = Path::new(&args[3]);
    let (name, mode, seed) = (&args[4], &args[5], args[6].parse()?);
    match args[0].as_str() {
        "train" | "train-scaled" => train_run::run(
            workspace,
            storage,
            name,
            mode,
            seed,
            &score::cell(cohort, name, mode, seed),
            protocol_text,
        ),
        "score" | "verify" | "score-scaled" | "verify-scaled" => score::run(
            workspace,
            storage,
            cohort,
            name,
            mode,
            seed,
            (protocol_text, command.starts_with("verify")),
        ),
        _ => Err("unknown command".into()),
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("REFUSED: {error}");
        std::process::exit(1);
    }
}

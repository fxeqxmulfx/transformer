//! Original object sequences, frozen training cohort and complete checkpoint replay.
use bilinear::experiment as original;
use convex_recurrent_sequence::{Result, experiment};
use serde_json::{Value, json};
use std::path::Path;

fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let text = experiment::PROTOCOL;
    let command = args.first().map_or("", String::as_str);
    if args.len() == 1 && command == "plan" {
        let protocol: Value = serde_json::from_str(text)?;
        for case in protocol["cases"].as_array().ok_or("cases")? {
            let name = case["name"].as_str().ok_or("case")?;
            let (_, row, task) = original::case_in(name, text)?;
            let model = experiment::initial_model(task.vocab, &protocol)?;
            let batch = row["case"]["batch"].as_u64().ok_or("batch")? as usize;
            println!(
                "{}",
                json!({"case":name,"objects":task.vocab,"budget":model.budget,"batch":batch,
                "training_examples":task.splits["train"].len(),"selection":task.selection,
                "real_coordinates_per_component":2*(1+2*task.vocab),
                "initial_reservation_bytes":model.reservation(batch)?,"source_fingerprint":experiment::fingerprint()})
            );
        }
        return Ok(());
    }
    if args.len() != 7 {
        return Err("usage: stress plan | stress train|score|verify WORKSPACE STORAGE COHORT CASE mini|forward SEED".into());
    }
    let workspace = Path::new(&args[1]);
    let storage = Path::new(&args[2]);
    let cohort = Path::new(&args[3]);
    let seed = args[6].parse()?;
    match command {
        "train" => experiment::train(workspace, storage, cohort, &args[4], &args[5], seed, text),
        "score" | "verify" => experiment::score::run(
            workspace,
            storage,
            cohort,
            &args[4],
            &args[5],
            seed,
            (text, command == "verify"),
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

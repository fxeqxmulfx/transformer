//! Native sealed-cohort training, complete original scoring and checkpoint replay.
use bilinear::experiment as original;
use convex_shared_atoms::{Result, experiment, model::Model};
use serde_json::{Value, json};
use std::path::Path;

fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() == 1 && args[0] == "plan" {
        let protocol: Value = serde_json::from_str(experiment::PROTOCOL)?;
        for case in protocol["cases"].as_array().ok_or("cases")? {
            let name = case["name"].as_str().ok_or("case")?;
            let (_, row, task) = original::case_in(name, experiment::PROTOCOL)?;
            let shape = original::shape(&task)?;
            let budget = protocol["budget_multiplier"].as_f64().ok_or("capacity")?
                * ((shape.objects as f64) * (shape.queries as f64) * (shape.columns as f64)).sqrt();
            let batch = row["case"]["batch"].as_u64().ok_or("batch")? as usize;
            println!(
                "{}",
                json!({"case":name,"shape":shape,"budget":budget,"batch":batch,
                "training_examples":task.splits["train"].len(),"selection":task.selection,
                "complete_real_coordinates":shape.real_coordinates().to_string(),
                "initial_reservation_bytes":Model::zero(shape,budget)?.reservation(batch)?,
                "source_fingerprint":experiment::fingerprint()})
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
    match args[0].as_str() {
        "train" => experiment::train(workspace, storage, cohort, &args[4], &args[5], seed),
        "score" | "verify" => experiment::score::run(
            workspace,
            storage,
            cohort,
            &args[4],
            &args[5],
            seed,
            args[0] == "verify",
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

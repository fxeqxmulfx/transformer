//! Native sealed-cohort training, complete original scoring and checkpoint replay.
use bilinear::experiment as original;
use convex_shared_atoms::{Result, experiment};
use serde_json::{Value, json};
use std::path::Path;

fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let balanced = args.first().is_some_and(|s| s.ends_with("-balanced"));
    let text = if balanced {
        experiment::BALANCED_PROTOCOL
    } else {
        experiment::PROTOCOL
    };
    let command = args.first().map_or("", |s| s.trim_end_matches("-balanced"));
    if args.len() == 1 && command == "plan" {
        let protocol: Value = serde_json::from_str(text)?;
        for case in protocol["cases"].as_array().ok_or("cases")? {
            let name = case["name"].as_str().ok_or("case")?;
            let (_, row, task) = original::case_in(name, text)?;
            let shape = original::shape(&task)?;
            let model = experiment::initial_model(shape, &protocol)?;
            let budget = model.budget;
            let batch = row["case"]["batch"].as_u64().ok_or("batch")? as usize;
            println!(
                "{}",
                json!({"case":name,"shape":shape,"budget":budget,"batch":batch,
                "training_examples":task.splits["train"].len(),"selection":task.selection,
                "complete_real_coordinates":shape.real_coordinates().to_string(),
                "geometry":protocol["geometry"],"raw_bias_response":model.query_scale[0]*model.context_scale[0],
                "query_mass":model.query_mass(protocol["query_step"].as_f64().ok_or("query step")?)?,
                "initial_reservation_bytes":model.reservation(batch)?,
                "source_fingerprint":experiment::fingerprint()})
            );
        }
        return Ok(());
    }
    if args.len() != 7 {
        return Err("usage: stress plan[-balanced] | stress train|score|verify[-balanced] WORKSPACE STORAGE COHORT CASE mini|forward SEED".into());
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

//! Frozen plain/learned-rotary comparison; no task identity enters model inference.
use bilinear::experiment as original;
use convex_rotary_recurrence::{Result, atom::Family, experiment};
use serde_json::{Value, json};
use std::path::Path;
fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let command = args.first().map_or("", String::as_str);
    let text = experiment::PROTOCOL;
    if args.len() == 1 && command == "plan" {
        let p: Value = serde_json::from_str(text)?;
        for case in p["cases"].as_array().ok_or("cases")? {
            let name = case["name"].as_str().ok_or("case")?;
            let (_, row, task) = original::case_in(name, text)?;
            for f in p["families"].as_array().ok_or("families")? {
                let family = Family::parse(f.as_str().ok_or("family")?)?;
                let model = experiment::initial_model(task.vocab, &p, family)?;
                let batch = row["case"]["batch"].as_u64().ok_or("batch")? as usize;
                println!(
                    "{}",
                    json!({"case":name,"family":family,"objects":task.vocab,
                    "budget":model.budget,"batch":batch,"training_examples":task.splits["train"].len(),
                    "selection":task.selection,"real_coordinates_per_component":family.coordinates(task.vocab),
                    "initial_reservation_bytes":model.reservation(batch)?,"source_fingerprint":experiment::fingerprint()})
                );
            }
        }
        return Ok(());
    }
    if args.len() != 8 {
        return Err("usage: stress plan | stress train|score|verify WORKSPACE STORAGE COHORT CASE plain|rotary mini|forward SEED".into());
    }
    let workspace = Path::new(&args[1]);
    let storage = Path::new(&args[2]);
    let cohort = Path::new(&args[3]);
    let family = Family::parse(&args[5])?;
    let seed = args[7].parse()?;
    match command {
        "train" => experiment::train(
            workspace,
            storage,
            cohort,
            &args[4],
            &args[6],
            seed,
            (text, family),
        ),
        "score" | "verify" => experiment::score::run(
            workspace,
            storage,
            cohort,
            &args[4],
            &args[6],
            seed,
            (text, command == "verify", family),
        ),
        _ => Err("unknown command".into()),
    }
}
fn main() {
    if let Err(e) = run() {
        eprintln!("REFUSED: {e}");
        std::process::exit(1);
    }
}

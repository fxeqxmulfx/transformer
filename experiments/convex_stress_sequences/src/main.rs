use convex_stress_sequences::{Result, dataset, runner};
use serde_json::Value;
use std::{fs::OpenOptions, io::Write, path::Path};

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [command, path] if command == "prepare" => {
            let catalog: Value = serde_json::from_str(convex_stress_sequences::CATALOG)?;
            let rows: Vec<_> = catalog["first_cohort"]
                .as_array()
                .unwrap()
                .iter()
                .map(|name| dataset::build(name.as_str().unwrap()).map(|data| data.metadata()))
                .collect::<Result<_>>()?;
            let mut out = OpenOptions::new().write(true).create_new(true).open(path)?;
            serde_json::to_writer_pretty(&mut out, &rows)?;
            out.write_all(b"\n")?;
            out.sync_all()?;
            println!("prepared4 original-specification sequence datasets; new model evaluations0");
            Ok(())
        }
        [command, root, producer, history] if command == "prepare-run" =>
            runner::prepare(Path::new(root), producer, Path::new(history)),
        [command, root, case] if command == "stage" => runner::stage(Path::new(root), case),
        [command, root, case] if command == "score" => runner::score(Path::new(root), case),
        [command, root, case] if command == "replay" => runner::replay(Path::new(root), case),
        [command, root, output] if command == "summary" =>
            runner::artifacts::write(Path::new(output), &runner::summary(Path::new(root))?),
        _ => Err("usage: sequences prepare FRESH_METADATA | prepare-run FRESH_ROOT PRODUCER HISTORY | {stage,score,replay} ROOT CASE | summary ROOT FRESH_OUTPUT".into()),
    }
}

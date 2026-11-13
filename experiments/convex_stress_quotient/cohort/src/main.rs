use convex_stress_quotient_cohort::{Configuration, artifacts, fingerprint, original, replay, run};
use std::{io::Write, path::Path};
fn main() -> convex_stress_quotient::Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["fingerprint"] => println!("{}", fingerprint()),
        ["prepare-original", request, root] => original::prepare(
            Path::new(root),
            artifacts::read::<original::Request>(Path::new(request))?,
        )?,
        ["run-original-case", root, task] => original::run_case(Path::new(root), task)?,
        ["replay-original-case", root, task] => {
            let value = original::replay_case(Path::new(root), task)?;
            serde_json::to_writer(std::io::stdout().lock(), &value)?;
            println!();
        }
        ["summarize-original", root] => {
            let value = original::summarize(Path::new(root))?;
            serde_json::to_writer(std::io::stdout().lock(), &value)?;
            println!();
        }
        ["run", config, root, producer] => run(
            artifacts::read::<Configuration>(Path::new(config))?,
            Path::new(root),
            producer,
        )?,
        ["replay", root] => {
            let value = replay(Path::new(root))?;
            let mut out = std::io::BufWriter::new(std::io::stdout().lock());
            serde_json::to_writer(&mut out, &value)?;
            writeln!(out)?;
        }
        _ => {
            return Err(
                "usage: quotient-cohort fingerprint | run CONFIG FRESH_ROOT PRODUCER | replay ROOT | prepare-original REQUEST FRESH_ROOT | run-original-case ROOT TASK | replay-original-case ROOT TASK | summarize-original ROOT"
                    .into(),
            );
        }
    }
    Ok(())
}

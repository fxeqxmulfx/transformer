use convex_stress_quotient_cohort::{Configuration, artifacts, fingerprint, replay, run};
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
                "usage: quotient-cohort fingerprint | run CONFIG FRESH_ROOT PRODUCER | replay ROOT"
                    .into(),
            );
        }
    }
    Ok(())
}

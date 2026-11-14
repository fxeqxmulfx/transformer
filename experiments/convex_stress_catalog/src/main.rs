use convex_stress_catalog::{Result, checked_text, inventory};
use std::{fs::OpenOptions, io::Write, path::Path};

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 5 {
        return Err(
            "usage: convex-stress-catalog CATALOG PLAN EXISTING35 HISTORY FRESH_OUTPUT".into(),
        );
    }
    let texts: Vec<_> = ["catalog", "plan", "existing35", "history"]
        .into_iter()
        .zip(&args)
        .map(|(name, path)| checked_text(name, Path::new(path)))
        .collect::<Result<_>>()?;
    let result = inventory(
        &texts[0],
        &serde_json::from_str(&texts[1])?,
        &serde_json::from_str(&texts[2])?,
        &serde_json::from_str(&texts[3])?,
    )?;
    let mut out = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[4])?;
    serde_json::to_writer_pretty(&mut out, &result)?;
    out.write_all(b"\n")?;
    out.sync_all()?;
    println!("165 cases:35 existing,130 additional; first cohort4; new model evaluations0");
    Ok(())
}

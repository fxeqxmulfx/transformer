//! Training-only development diagnostic; no heldout pools or scores.
use convex_stress_prefix_quotients::{Result, prior, search};
use std::{
    fs::OpenOptions,
    io::{BufWriter, Write},
};
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let [case, output] = args.as_slice() else {
        return Err("usage: discover CASE FRESH_JSON".into());
    };
    let input = prior::adapter::training_only(case)?;
    let report = search::run(
        &input.observations,
        input.vocabulary,
        search::Policy::default(),
    )?;
    let mut file = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output)?,
    );
    serde_json::to_writer(&mut file, &report)?;
    file.write_all(b"\n")?;
    file.flush()?;
    eprintln!(
        "{}: {} records, {} closed prefixes, {} compatible programs, {:?}",
        case,
        report.observed_records,
        report.closure_nodes,
        report.programs.len(),
        report.refusal
    );
    for s in report.stages {
        eprintln!("{}", serde_json::to_string(&s)?);
    }
    Ok(())
}

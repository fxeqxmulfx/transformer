use convex_consequences_cohort as cohort;
use convex_stress_consequences::Result;
use std::{io::Write, path::Path};
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [cmd] if cmd == "fingerprint" => println!("{}", cohort::fingerprint()),
        [cmd, root, parent] if cmd == "prepare" => cohort::prepare(Path::new(root), Path::new(parent))?,
        [cmd, root, task] if cmd == "run-cell" => cohort::run(Path::new(root), task)?,
        [cmd, root, task] if cmd == "replay-cell" => println!("{}", cohort::replay(Path::new(root), task)?),
        [cmd, root] if cmd == "summarize" => {
            let mut out = std::io::BufWriter::new(std::io::stdout().lock());
            serde_json::to_writer(&mut out, &cohort::summarize(Path::new(root))?)?;
            writeln!(out)?;
        },
        _ => return Err("usage: convex-consequences-cohort fingerprint|prepare ROOT IDENTIFIED_PARENT_ROOT|run-cell ROOT TASK|replay-cell ROOT TASK|summarize ROOT".into()),
    }
    Ok(())
}

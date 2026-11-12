use convex_stress_decode::{Result, cohort};
use std::path::Path;
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        ["fingerprint"] => println!("{}", cohort::fingerprint()),
        ["prepare", root, parent] => cohort::prepare(Path::new(root), Path::new(parent))?,
        ["run-cell", root, id] => cohort::run(Path::new(root), id)?,
        ["replay-cell", root, id] => println!("{}", cohort::replay(Path::new(root), id)?),
        ["summarize", root] => println!("{}", cohort::summarize(Path::new(root))?),
        _ => return Err("usage: convex-stress-decode fingerprint | prepare ROOT PHASE_ROOT | run-cell ROOT ID | replay-cell ROOT ID | summarize ROOT".into()),
    }
    Ok(())
}

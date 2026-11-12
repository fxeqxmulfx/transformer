use convex_stress_paired::{Result, cohort};
use std::path::Path;
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        ["fingerprint"] => println!("{}", cohort::fingerprint()),
        ["prepare", root] => cohort::prepare(Path::new(root))?,
        ["run-cell", root, id] => cohort::run(Path::new(root), id)?,
        ["replay-cell", root, id] => println!("{}", cohort::replay(Path::new(root), id)?),
        ["summarize", root] => println!("{}", cohort::summarize(Path::new(root))?),
        ["integrate", root, parent] => {
            println!("{}", cohort::integrate(Path::new(root), Path::new(parent))?)
        }
        _ => return Err("usage: convex-stress-paired fingerprint | prepare ROOT | run-cell ROOT ID | replay-cell ROOT ID | summarize ROOT | integrate ROOT DECODE_ROOT".into()),
    }
    Ok(())
}

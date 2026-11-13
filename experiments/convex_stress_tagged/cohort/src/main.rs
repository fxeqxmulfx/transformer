mod cohort;
use convex_stress_tagged::Result;
use std::path::Path;
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().skip(1).collect();
    match a.as_slice() {
        [cmd] if cmd == "fingerprint" => println!("{}", cohort::fingerprint()),
        [cmd, root] if cmd == "prepare" => cohort::prepare(Path::new(root))?,
        [cmd, root, task] if cmd == "run-cell" => cohort::run(Path::new(root), task)?,
        [cmd, root, task] if cmd == "replay-cell" => println!("{}", cohort::replay(Path::new(root), task)?),
        [cmd, root] if cmd == "summarize" => println!("{}", cohort::summarize(Path::new(root))?),
        _ => return Err("usage: convex-tagged-cohort fingerprint|prepare ROOT|run-cell ROOT TASK|replay-cell ROOT TASK|summarize ROOT".into()),
    }
    Ok(())
}

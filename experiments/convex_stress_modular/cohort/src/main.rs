mod cohort;
use convex_stress_modular::Result;
use std::path::Path;
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [cmd] if cmd=="fingerprint"=>println!("{}",cohort::fingerprint()),
        [cmd,root,parent] if cmd=="prepare"=>cohort::prepare(Path::new(root),Path::new(parent))?,
        [cmd,root,task] if cmd=="run-cell"=>cohort::run(Path::new(root),task)?,
        [cmd,root,task] if cmd=="replay-cell"=>println!("{}",cohort::replay(Path::new(root),task)?),
        [cmd,root] if cmd=="summarize"=>println!("{}",cohort::summarize(Path::new(root))?),
        _=>return Err("usage: convex-modular-cohort fingerprint|prepare ROOT PARENT_JSON|run-cell ROOT TASK|replay-cell ROOT TASK|summarize ROOT".into())
    }
    Ok(())
}

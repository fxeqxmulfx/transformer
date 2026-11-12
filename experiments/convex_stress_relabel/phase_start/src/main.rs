use convex_phase_start::cohort;
use convex_stress_relabel::Result;
use std::path::Path;
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        ["fingerprint"]=>println!("{}",cohort::fingerprint()),
        ["prepare",root,cold,graph]=>cohort::prepare(Path::new(root),Path::new(cold),Path::new(graph))?,
        ["run-cell",root,id]=>cohort::run(Path::new(root),id)?,
        ["replay-cell",root,id]=>println!("{}",cohort::replay(Path::new(root),id)?),
        ["summarize",root]=>println!("{}",cohort::summarize(Path::new(root))?),
        _=>return Err("usage: convex-phase-start fingerprint | prepare ROOT COLD_ROOT GRAPH_ROOT | run-cell ROOT ID | replay-cell ROOT ID | summarize ROOT".into()),
    }
    Ok(())
}

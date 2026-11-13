use convex_stress_causal_pairs_cohort::{
    Request, artifacts, fingerprint, prepare, replay_case, run_case, summarize,
};
use std::path::Path;
fn main() -> convex_stress_causal_pairs::Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        ["fingerprint"] => println!("{}", fingerprint()),
        ["prepare-original", request, root] => prepare(Path::new(root), artifacts::read::<Request>(Path::new(request))?)?,
        ["run-original-case", root, task] => run_case(Path::new(root), task)?,
        ["replay-original-case", root, task] => { serde_json::to_writer(std::io::stdout().lock(), &replay_case(Path::new(root), task)?)?; println!(); }
        ["summarize-original", root] => { serde_json::to_writer(std::io::stdout().lock(), &summarize(Path::new(root))?)?; println!(); }
        _ => return Err("usage: causal-pairs-cohort fingerprint | prepare-original REQUEST FRESH_ROOT | run-original-case ROOT TASK | replay-original-case ROOT TASK | summarize-original ROOT".into()),
    }
    Ok(())
}

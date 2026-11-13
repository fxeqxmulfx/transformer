use convex_stress_causal_pairs_cohort::{
    Request, artifacts, fingerprint, prepare, replay_case, run_case, staging, summarize,
};
use std::path::Path;
fn main() -> convex_stress_causal_pairs::Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        ["fingerprint"] => println!("{}", fingerprint()),
        ["prepare-training", request, root] => staging::prepare(Path::new(root), artifacts::read::<staging::Request>(Path::new(request))?)?,
        ["run-training-case", root, task] => staging::run_case(Path::new(root), task)?,
        ["replay-training-case", root, task] => { serde_json::to_writer(std::io::stdout().lock(), &staging::replay_case(Path::new(root), task)?)?; println!(); }
        ["summarize-training", root] => { serde_json::to_writer(std::io::stdout().lock(), &staging::summarize(Path::new(root))?)?; println!(); }
        ["prepare-original", request, root] => prepare(Path::new(root), artifacts::read::<Request>(Path::new(request))?)?,
        ["run-original-case", root, task] => run_case(Path::new(root), task)?,
        ["replay-original-case", root, task] => { serde_json::to_writer(std::io::stdout().lock(), &replay_case(Path::new(root), task)?)?; println!(); }
        ["summarize-original", root] => { serde_json::to_writer(std::io::stdout().lock(), &summarize(Path::new(root))?)?; println!(); }
        _ => return Err("usage: causal-pairs-cohort fingerprint | prepare-{training,original} REQUEST FRESH_ROOT | {run,replay}-{training,original}-case ROOT TASK | summarize-{training,original} ROOT".into()),
    }
    Ok(())
}

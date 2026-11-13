use convex_stress_quotient_cohort::artifacts;
use convex_stress_refined_quotient_cohort::{
    Request, fingerprint, prepare, replay_case, run_case, summarize,
};
use std::path::Path;
fn main() -> convex_stress_refined_quotient::Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        ["fingerprint"] => println!("{}",fingerprint()),
        ["prepare", request, root] => prepare(Path::new(root),artifacts::read::<Request>(Path::new(request))?)?,
        ["run-case", root, task] => run_case(Path::new(root),task)?,
        ["replay-case", root, task] => { serde_json::to_writer(std::io::stdout().lock(),&replay_case(Path::new(root),task)?)?; println!(); },
        ["summarize", root] => { serde_json::to_writer(std::io::stdout().lock(),&summarize(Path::new(root))?)?; println!(); },
        _ => return Err("usage: refined-cohort fingerprint | prepare REQUEST FRESH_ROOT | run-case ROOT TASK | replay-case ROOT TASK | summarize ROOT".into()),
    }
    Ok(())
}

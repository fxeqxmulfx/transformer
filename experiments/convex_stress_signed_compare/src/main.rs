use convex_stress_signed_compare::{self as runner, Request};
use std::{fs, path::Path};
fn main() -> convex_stress_signed_kernel::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    match a.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        ["fingerprint"] => println!("{}",runner::fingerprint()),
        ["forecast",inspection] => println!("{}",serde_json::to_string(&runner::forecast_inspection(Path::new(inspection))?)?),
        ["prepare",root,request] => runner::prepare(Path::new(root),serde_json::from_slice::<Request>(&fs::read(request)?)?)?,
        ["run-case",root,task] => runner::run_case(Path::new(root),task)?,
        ["replay-case",root,task] => println!("{}",serde_json::to_string(&runner::replay_case(Path::new(root),task)?)?),
        ["summarize",root] => println!("{}",serde_json::to_string(&runner::summarize(Path::new(root))?)?),
        _ => return Err("usage: fingerprint | forecast INSPECTION | prepare ROOT REQUEST | run-case ROOT TASK | replay-case ROOT TASK | summarize ROOT".into()),
    }
    Ok(())
}

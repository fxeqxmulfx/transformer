use convex_stress_shared_compare as app;
use std::{env, path::Path};
fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    match args.iter().map(String::as_str).collect::<Vec<_>>().as_slice(){
        ["fingerprint"]=>println!("{}",app::fingerprint()),
        ["prepare",request,root]=>app::prepare(Path::new(root),serde_json::from_slice(&std::fs::read(request)?)?)?,
        ["run-case",root,task]=>app::run_case(Path::new(root),task)?,
        ["replay-case",root,task]=>println!("{}",serde_json::to_string_pretty(&app::replay_case(Path::new(root),task)?)?),
        ["summarize",root]=>println!("{}",serde_json::to_string_pretty(&app::summarize(Path::new(root))?)?),
        _=>return Err("usage: fingerprint | prepare REQUEST FRESH_ROOT | run-case ROOT TASK | replay-case ROOT TASK | summarize ROOT".into())
    }
    Ok(())
}

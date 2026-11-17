use convex_retained_readouts::{Result, diagnostic, experiment, prior, score, train};
use std::path::Path;
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    match a.get(1).map(String::as_str) {
        Some("plan") if a.len()==2 => { let p:serde_json::Value=serde_json::from_str(experiment::PROTOCOL)?;
            println!("{}",serde_json::to_string_pretty(&serde_json::json!({"source_fingerprint":experiment::fingerprint(),"protocol":p}))?); Ok(()) }
        Some("train") if a.len()==9 => train::run(train::Paths{workspace:Path::new(&a[2]),storage:Path::new(&a[3]),
            sources:Path::new(&a[4]),cohort:Path::new(&a[5])},&a[6],&a[7],&a[8]),
        Some(cmd @ ("score"|"replay")) if a.len()==8 => score::run(Path::new(&a[2]),Path::new(&a[3]),Path::new(&a[4]),&a[5],&a[6],&a[7],cmd=="replay"),
        Some("diagnose") if a.len()==6 => diagnostic::run(Path::new(&a[2]),Path::new(&a[3]),Path::new(&a[4]),Path::new(&a[5])),
        Some("prior") if a.len()==6 => prior::run(Path::new(&a[2]),Path::new(&a[3]),Path::new(&a[4]),Path::new(&a[5])),
        _ => Err("usage: convex-retained-readouts plan | train WORKSPACE STORAGE SOURCES COHORT ARM VARIANT MODE | score/replay WORKSPACE STORAGE COHORT ARM VARIANT MODE | diagnose/prior WORKSPACE STORAGE COHORT NEW_OUTPUT".into()),
    }
}

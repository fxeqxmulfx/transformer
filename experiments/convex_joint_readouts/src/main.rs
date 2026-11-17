use convex_joint_readouts::{Result, diagnose, experiment, score, train};
use std::path::Path;
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("plan") if args.len()==2 => {
            let p:serde_json::Value=serde_json::from_str(experiment::PROTOCOL)?;
            println!("{}",serde_json::to_string_pretty(&serde_json::json!({
                "source_fingerprint":experiment::fingerprint(),"protocol":p}))?); Ok(())
        }
        Some("diagnose") if args.len()==6 => diagnose::run(Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4]),Path::new(&args[5])),
        Some("train") if args.len()==8 => train::run(Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4]),Path::new(&args[5]),&args[6],&args[7]),
        Some(cmd @ ("score"|"replay")) if args.len()==7 => score::run(Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4]),&args[5],&args[6],cmd=="replay"),
        _ => Err("usage: convex-joint-readouts plan | diagnose WORKSPACE STORAGE SOURCES NEW_OUTPUT | train WORKSPACE STORAGE SOURCES COHORT VARIANT MODE | score/replay WORKSPACE STORAGE COHORT VARIANT MODE".into()),
    }
}

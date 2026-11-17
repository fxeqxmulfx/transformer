use convex_readout_optimality::{Result, experiment};
use std::path::Path;
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("plan") if args.len()==2=>{
            let p:serde_json::Value=serde_json::from_str(experiment::PROTOCOL)?;
            println!("{}",serde_json::to_string_pretty(&serde_json::json!({
                "source_fingerprint":experiment::fingerprint(),"protocol":p}))?);
            Ok(())
        }
        Some("run") if args.len()==6=>experiment::run(Path::new(&args[2]),Path::new(&args[3]),
            Path::new(&args[4]),Path::new(&args[5])),
        _=>Err("usage: convex-readout-optimality plan | run WORKSPACE STORAGE ROTARY_SOURCES NEW_OUTPUT".into()),
    }
}

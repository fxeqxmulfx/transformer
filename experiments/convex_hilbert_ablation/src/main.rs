use convex_hilbert_ablation::{Result, experiment, source};
use std::path::Path;
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    match a.get(1).map(String::as_str) {
        Some("plan") if a.len()==2=>{let p:serde_json::Value=serde_json::from_str(experiment::PROTOCOL)?;println!("{}",serde_json::to_string_pretty(&serde_json::json!({"source_fingerprint":experiment::fingerprint(),"protocol":p}))?);Ok(())},
        Some("run") if a.len()==6=>experiment::run(Path::new(&a[2]),Path::new(&a[3]),Path::new(&a[4]),Path::new(&a[5])),
        Some("verify") if a.len()==5=>{source::verify(Path::new(&a[2]),Path::new(&a[3]),Path::new(&a[4]))?;println!("all frozen ablation inputs match");Ok(())},
        _=>Err("usage: convex-hilbert-ablation plan | run WORKSPACE STORAGE COHORT NEW_OUTPUT | verify WORKSPACE STORAGE COHORT".into())
    }
}

use convex_content_growth::{Result, diagnostic, experiment, score, train};
use std::path::Path;
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    match a.get(1).map(String::as_str){
        Some("plan") if a.len()==2=>{println!("{}",serde_json::to_string_pretty(&serde_json::json!({"source_fingerprint":experiment::fingerprint(),"protocol":experiment::protocol()?}))?);Ok(())},
        Some("train") if a.len()==9=>train::run(Path::new(&a[2]),Path::new(&a[3]),Path::new(&a[4]),&experiment::Cell::new(&a[5],&a[6],&a[7],&a[8])?),
        Some(cmd@("score"|"replay")) if a.len()==5=>score::run(Path::new(&a[2]),Path::new(&a[3]),Path::new(&a[4]),cmd=="replay"),
        Some("diagnose") if a.len()==6=>diagnostic::run(Path::new(&a[2]),Path::new(&a[3]),Path::new(&a[4]),Path::new(&a[5])),
        _=>Err("usage: convex-content-growth plan | train WORKSPACE STORAGE COHORT ARM GEOMETRY VARIANT MODE | score/replay WORKSPACE STORAGE COHORT | diagnose WORKSPACE STORAGE COHORT NEW_OUTPUT".into())
    }
}

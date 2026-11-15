use convex_stress_remaining_synthetic_data::{Result, inventory};
use std::path::Path;
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [cmd,history,output] if cmd=="check-history"=>inventory::check_history(Path::new(history),Path::new(output)),
        [cmd,history,baseline,root,producer] if cmd=="prepare"=>
            inventory::prepare(Path::new(history),Path::new(baseline),Path::new(root),producer),
        _=>Err("usage: remaining-synthetic-data check-history HISTORY FRESH_OUTPUT | prepare HISTORY REVIEWED339_ROOT FRESH_ROOT COMMIT".into())
    }
}

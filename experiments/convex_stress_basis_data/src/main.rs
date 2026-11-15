use convex_stress_basis_data::{Result, inventory};
use std::path::Path;
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [cmd,history,output] if cmd=="check-history"=>inventory::check_history(Path::new(history),Path::new(output)),
        [cmd,history,baseline,root,producer] if cmd=="prepare"=>inventory::prepare(Path::new(history),Path::new(baseline),Path::new(root),producer),
        [cmd,original,root] if cmd=="repeat"=>inventory::repeat(Path::new(original),Path::new(root)),
        _=>Err("usage: basis-data check-history HISTORY FRESH_JSON | prepare HISTORY BASELINE344_ROOT FRESH_ROOT COMMIT | repeat ORIGINAL_ROOT FRESH_ROOT".into())
    }
}

use convex_stress_generative_data::{Result, inventory};
use std::path::Path;
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [command, history, baseline, output] if command == "prepare" => {
            inventory::prepare(Path::new(history), Path::new(baseline), Path::new(output))
        }
        _ => Err("usage: generative-data prepare FROZEN_HISTORY BASELINE326 FRESH_OUTPUT".into()),
    }
}

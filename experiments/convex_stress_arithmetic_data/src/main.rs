use convex_stress_arithmetic_data::{Result, inventory};
use std::path::Path;
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [cmd, input, output, producer] if cmd == "prepare" => inventory::prepare(Path::new(input),Path::new(output),producer),
        [cmd, original, output] if cmd == "repeat" => inventory::repeat(Path::new(original),Path::new(output)),
        [cmd, original, repeated, output] if cmd == "verify" => inventory::verify(Path::new(original),Path::new(repeated),Path::new(output)),
        _ => Err("usage: arithmetic-data prepare INPUTS FRESH_ROOT COMMIT | repeat ORIGINAL FRESH_ROOT | verify ORIGINAL REPEAT FRESH_JSON".into()),
    }
}

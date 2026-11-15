use convex_stress_generative_programs::{Result, runner};
use std::path::Path;
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [command, source, root] if command == "repeat-run" =>
            runner::repeat(Path::new(source),Path::new(root)),
        [command, root, producer, baseline, previous] if command == "prepare-run" =>
            runner::prepare(Path::new(root), producer, Path::new(baseline),Path::new(previous)),
        [command, root, case] if command == "stage" => runner::stage(Path::new(root), case),
        [command, root, case] if command == "score" => runner::score(Path::new(root), case),
        [command, root, case] if command == "replay" => runner::replay(Path::new(root), case),
        [command, root, output] if command == "summary" =>
            runner::artifacts::write(Path::new(output), &runner::summary(Path::new(root))?),
        _ => Err("usage: generative-programs prepare-run FRESH_ROOT PRODUCER BASELINE_INVENTORY PREVIOUS_REPORT | repeat-run SOURCE FRESH_ROOT | {stage,score,replay} ROOT CASE | summary ROOT FRESH_OUTPUT".into()),
    }
}

mod cohort;
mod summary;
use convex_stress_calibrated::Result;
use std::path::Path;
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [cmd] if cmd == "fingerprint" => { println!("{}", cohort::fingerprint()); Ok(()) },
        [cmd] if cmd == "prepare" => {
            for row in cohort::rows()? {
                let task = convex_stress_calibrated::data::build(&row)?;
                println!("verified {} {} splits", row["name"], task.splits.len());
            }
            Ok(())
        },
        [cmd, arm, task, out] if cmd == "run" => cohort::run(arm, task, Path::new(out)),
        [cmd, out] if cmd == "replay" => cohort::replay(Path::new(out)),
        [cmd, out] if cmd == "summary" => summary::summarize(Path::new(out)),
        _ => Err("usage: fingerprint | prepare | run READOUT TASK FRESH_DIR | replay DIR | summary COHORT".into()),
    }
}

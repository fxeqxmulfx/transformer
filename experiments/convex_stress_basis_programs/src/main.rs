use convex_stress_basis_programs::*;
use std::path::Path;

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [command,data,root,producer] if command=="prepare"=>runner::prepare(Path::new(data),Path::new(root),producer)?,
        [command,root] if command=="train"=>runner::train(Path::new(root))?,
        [command,root] if command=="score"=>runner::score(Path::new(root))?,
        [command,root] if command=="summary"=>runner::summary(Path::new(root))?,
        [command,source,root] if command=="repeat"=>runner::repeat(Path::new(source),Path::new(root))?,
        [command,source,review,output] if command=="verify-pair"=>runner::verify_pair(Path::new(source),Path::new(review),Path::new(output))?,
        [command, case, path] if command == "discover" => {
            let (v, rows) = input::training_only(case)?;
            let discovery = discovery::run(v, &rows)?;
            artifacts::write(Path::new(path), &discovery)?;
            println!(
                "training records={} candidates={} retained={}",
                discovery.input.records,
                discovery.attempts.len(),
                usize::from(discovery.selected.is_some())
            );
        }
        [command, case, path] if command == "fit-training" => {
            let (v, rows) = input::training_only(case)?;
            let stage = fit::construct(v, &rows)?;
            artifacts::write(Path::new(path), &stage)?;
            println!(
                "training records={} candidates={} fitted={}/6",
                stage.discovery.input.records,
                stage.discovery.attempts.len(),
                stage.endpoints.iter().filter(|e| e.error.is_none()).count()
            );
        }
        _=>return Err("commands: discover|fit-training CASE FRESH_JSON; prepare DATA345_ROOT FRESH_ROOT COMMIT; train|score|summary ROOT; repeat ORIGINAL FRESH_ROOT; verify-pair ORIGINAL REVIEW FRESH_JSON".into()),
    }
    Ok(())
}

use composed_retrieval::{Result, cohort::Run, evaluation};
use std::path::Path;
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let [action, workspace, storage, out, arm, rest @ ..] = args.as_slice() else {
        return Err("expected action workspace storage output arm [case [mode seed]]".into());
    };
    let run = Run::new(
        Path::new(workspace),
        Path::new(storage),
        Path::new(out),
        arm,
    )?;
    match (action.as_str(), rest) {
        ("fit", [id, mode, seed]) => run.fit(id, mode, seed.parse()?),
        ("seal", []) => run.seal(),
        ("score", [id]) => evaluation::score(&run, id, false),
        ("replay", [id]) => evaluation::score(&run, id, true),
        _ => Err("unknown action or arguments".into()),
    }
}

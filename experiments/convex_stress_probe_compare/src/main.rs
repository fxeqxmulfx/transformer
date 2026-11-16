use convex_stress_probe_compare::{
    Result, inputs, resources::Guard, scoring, seal, stage, verification,
};
use std::path::Path;
fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let words: Vec<&str> = args.iter().map(String::as_str).collect();
    let workspace = words.get(1).ok_or("expected command and workspace")?;
    let root = Path::new(words.get(2).ok_or("expected output or original path")?);
    let parent = root.parent().ok_or("missing output parent")?;
    Guard::new(Path::new(workspace), parent)?.watch()?;
    match words.as_slice() {
        ["prepare", workspace, root, data, producer] => inputs::prepare(
            Path::new(workspace),
            Path::new(root),
            Path::new(data),
            producer,
        )?,
        ["stage", workspace, root, case] => {
            stage::run(Path::new(workspace), Path::new(root), case)?
        }
        ["seal", workspace, root] => seal::create(Path::new(workspace), Path::new(root))?,
        ["score", workspace, root, case] => {
            scoring::run(Path::new(workspace), Path::new(root), case, false)?
        }
        ["replay", workspace, root, case] => {
            scoring::run(Path::new(workspace), Path::new(root), case, true)?
        }
        ["verify", workspace, original, repeat, output] => verification::verify(
            Path::new(workspace),
            Path::new(original),
            Path::new(repeat),
            Path::new(output),
        )?,
        _ => {
            return Err(
                "expected prepare, stage, seal, score, replay or verify with all explicit paths"
                    .into(),
            );
        }
    }
    Ok(())
}

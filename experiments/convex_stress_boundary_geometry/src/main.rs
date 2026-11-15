use convex_stress_boundary_geometry::{Mode, Result, inspection};
use std::path::Path;
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if let [command, parent, sha, role, mode, out] = args.as_slice()
        && command == "inspect"
    {
        let mode = match mode.as_str() {
            "Full" => Mode::Full,
            "Mini" => Mode::Mini,
            "Forward" => Mode::Forward,
            _ => return Err("invalid mode".into()),
        };
        return inspection::run(Path::new(parent), sha, role, mode, Path::new(out));
    }
    Err(
        "usage: inspect COMPLETE_PARENT SHA256 prior|supplement Full|Mini|Forward FRESH_ROOT"
            .into(),
    )
}

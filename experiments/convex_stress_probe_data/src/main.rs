use convex_stress_probe_data::{Result, cases, dataset, inventory};
use std::path::Path;
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [command, workspace, output, helper, producer] if command == "prepare" => {
            if !inventory::prepare(Path::new(workspace), Path::new(output), Path::new(helper), producer)? {
                return Err("inventory retained: at least one original dataset remains unverified".into());
            }
        }
        [command, workspace, original, repeat, output] if command == "verify" => {
            inventory::verify(Path::new(workspace), Path::new(original), Path::new(repeat), Path::new(output))?;
        }
        [command, helper] if command == "inspect" => {
            let mut failed = false;
            for source in cases()? {
                match dataset::build(&source, Path::new(helper)) {
                    Ok(data) => println!("{}", serde_json::to_string(&data.summary()?)?),
                    Err(error) => {
                        eprintln!("{}: {error}", source["case"]["name"]);
                        failed = true;
                    }
                }
            }
            if failed {
                return Err("complete probe inspection failed".into());
            }
        }
        _ => return Err("expected inspect NATIVE_REFERENCE | prepare WORKSPACE NEW_OUTPUT NATIVE_REFERENCE PRODUCER | verify WORKSPACE ORIGINAL REPEAT NEW_REPORT".into()),
    }
    Ok(())
}

use convex_stress_transformer::{Result, cohort, preflight};
use std::path::Path;
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [command, task] if command == "cohort-preflight" => println!("{}", cohort::preflight(task)?),
        [command] if command == "fingerprint" => println!("{}",cohort::fingerprint()),
        [command,root,parent] if command == "prepare" => cohort::prepare(Path::new(root),Path::new(parent))?,
        [command,root,id] if command == "run-cell" => cohort::run(Path::new(root),id)?,
        [command,root,id] if command == "replay-cell" => println!("{}",cohort::replay(Path::new(root),id)?),
        [command,root] if command == "summarize" => println!("{}",cohort::summarize(Path::new(root))?),
        [command] if command == "tasks" => {
            for row in preflight::rows()? {
                println!("{}", row["name"].as_str().ok_or("invalid task name")?);
            }
        }
        [command, task] if command == "preflight" => {
            let row = preflight::rows()?
                .into_iter()
                .find(|r| r["name"] == *task)
                .ok_or("unknown original task")?;
            println!(
                "{}",
                serde_json::to_string_pretty(&preflight::inspect(&row)?)?
            );
        }
        _ => return Err("usage: convex-stress-transformer tasks | preflight ORIGINAL_TASK | cohort-preflight ORIGINAL_TASK | fingerprint | prepare ROOT DECODE_ROOT | run-cell ROOT ID | replay-cell ROOT ID | summarize ROOT".into()),
    }
    Ok(())
}

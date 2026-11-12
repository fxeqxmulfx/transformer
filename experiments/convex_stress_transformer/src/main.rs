use convex_stress_transformer::{Result, preflight};
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.as_slice() {
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
        _ => return Err("usage: convex-stress-transformer tasks | preflight ORIGINAL_TASK".into()),
    }
    Ok(())
}

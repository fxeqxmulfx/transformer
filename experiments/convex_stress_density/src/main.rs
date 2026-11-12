use convex_stress_density::{Result, artifact};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
};

fn read<T: serde::de::DeserializeOwned>(path: &str, limit: u64) -> Result<T> {
    if fs::metadata(path)?.len() > limit {
        return Err("artifact exceeds declared file-size bound".into());
    }
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 3 || !matches!(args[0].as_str(), "extract" | "replay") {
        return Err("usage: convex-stress-density extract|replay INPUT.json ARTIFACT.json".into());
    }
    let input: artifact::Input = read(&args[1], 4 * 1024 * 1024)?;
    if args[0] == "replay" {
        let saved = read(&args[2], 64 * 1024 * 1024)?;
        artifact::replay(&input, &saved)?;
        println!("extraction replay and raw audit passed");
    } else {
        if Path::new(&args[2]).exists() {
            return Err("artifact destination already exists".into());
        }
        let out = artifact::run(input)?;
        let bytes = serde_json::to_vec_pretty(&out)?;
        if bytes.len() > 64 * 1024 * 1024 {
            return Err("output exceeds artifact byte budget".into());
        }
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&args[2])?;
        file.write_all(&bytes)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        println!(
            "accepted={} reason={} stages={} iterations={}",
            out.extraction.accepted,
            out.extraction.reason,
            out.extraction.stages.len(),
            out.extraction
                .stages
                .iter()
                .map(|s| s.trace.len())
                .sum::<usize>()
        );
    }
    Ok(())
}

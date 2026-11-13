use convex_stress_density::{Result, artifact, interval, interval_audit, interval_search, search};
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
    if args.len() != 3
        || !matches!(
            args[0].as_str(),
            "extract"
                | "replay"
                | "propose"
                | "replay-proposal"
                | "interval"
                | "replay-interval"
                | "propose-interval"
                | "replay-interval-proposal"
        )
    {
        return Err("usage: convex-stress-density extract|replay|propose|replay-proposal|interval|replay-interval|propose-interval|replay-interval-proposal INPUT.json ARTIFACT.json".into());
    }
    if args[0] == "propose-interval" || args[0] == "replay-interval-proposal" {
        let input: interval_search::Input = read(&args[1], 4 * 1024 * 1024)?;
        if args[0] == "replay-interval-proposal" {
            let saved = read(&args[2], 256 * 1024 * 1024)?;
            interval_search::replay(&input, &saved)?;
            println!("exact and interval proposals, endpoints and raw audits replayed");
        } else {
            if Path::new(&args[2]).exists() {
                return Err("interval proposal destination already exists".into());
            }
            let out = interval_search::propose(input)?;
            write_new(&args[2], &out, 256 * 1024 * 1024)?;
            println!(
                "attempts={} selected={:?} iterations={}",
                out.attempts.len(),
                out.selected,
                out.cost.iterations_charged
            );
        }
        return Ok(());
    }
    if args[0] == "interval" || args[0] == "replay-interval" {
        let input: interval::Input = read(&args[1], 4 * 1024 * 1024)?;
        if args[0] == "replay-interval" {
            let saved = read(&args[2], 64 * 1024 * 1024)?;
            interval_audit::replay(&input, &saved)?;
            println!("interval density state and raw support-box bound replayed");
        } else {
            if Path::new(&args[2]).exists() {
                return Err("interval destination already exists".into());
            }
            let out = interval_audit::run(input)?;
            write_new(&args[2], &out, 64 * 1024 * 1024)?;
            println!(
                "accepted={} iterations={} purity_error={}",
                out.outcome.accepted,
                out.outcome.trace.len(),
                out.outcome.purity_error
            );
        }
        return Ok(());
    }
    if args[0] == "propose" || args[0] == "replay-proposal" {
        let input: search::Input = read(&args[1], 4 * 1024 * 1024)?;
        if args[0] == "replay-proposal" {
            let saved = read(&args[2], 256 * 1024 * 1024)?;
            search::replay(&input, &saved)?;
            println!("categorical proposal, phase endpoint and raw audits replayed");
        } else {
            if Path::new(&args[2]).exists() {
                return Err("proposal destination already exists".into());
            }
            let out = search::propose(input)?;
            write_new(&args[2], &out, 256 * 1024 * 1024)?;
            println!(
                "attempts={} selected={:?} iterations={} charged={}",
                out.attempts.len(),
                out.selected,
                out.cost.admm_iterations,
                out.cost.iteration_budget_used
            );
        }
        return Ok(());
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
        write_new(&args[2], &out, 64 * 1024 * 1024)?;
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

fn write_new(path: &str, value: &impl serde::Serialize, limit: usize) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value)?;
    if bytes.len() > limit {
        return Err("output exceeds artifact byte budget".into());
    }
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(&bytes)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(())
}

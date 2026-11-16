//! Training-only boundary. Metadata and heldout files never enter this process.
use convex_stress_wide_latents::*;
use serde::Serialize;
use std::{
    fs::File,
    io::{BufReader, Write},
    path::Path,
};

#[derive(Serialize)]
struct Report<'a> {
    selection: &'a selection::Selection,
    work: &'a Work,
    protocol_sha256: String,
    expanded_training_records_allocated: usize,
    heldout_pools_read: usize,
    accuracy_measurements: usize,
    training_updates: usize,
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if let [command, workspace, storage] = args.as_slice()
        && command == "check"
    {
        joint::resources::Guard::new(Path::new(workspace), Path::new(storage), COHORT_RESERVE)?
            .check()?;
        return Ok(());
    }
    let [command, input, output, workspace, storage] = args.as_slice() else {
        return Err("usage: convex-stress-wide-latents probe TRAINING_JSON NEW_OUTPUT WORKSPACE STORAGE | check WORKSPACE STORAGE".into());
    };
    if command != "probe" {
        return Err("unknown training-only command".into());
    }
    if Path::new(output).try_exists()? {
        return Err("refusing to replace an existing probe report".into());
    }
    let guard =
        joint::resources::Guard::new(Path::new(workspace), Path::new(storage), COHORT_RESERVE)?;
    let file = File::open(input)?;
    if file.metadata()?.len() > 8 * 1024 * 1024 {
        return Err("training JSON exceeds8MiB".into());
    }
    let training: Training = serde_json::from_reader(BufReader::new(file))?;
    training.validate()?;
    let mut work = Work::new(MAX_WORK)?;
    let selection = selection::select(&training, &mut work)?;
    let report = Report {
        selection: &selection,
        work: &work,
        protocol_sha256: digest(PROTOCOL.as_bytes()),
        expanded_training_records_allocated: 0,
        heldout_pools_read: 0,
        accuracy_measurements: 0,
        training_updates: 0,
    };
    let bytes = serde_json::to_vec(&report)?;
    if bytes.len() as u64 > REPORT_LIMIT {
        return Err("complete report exceeds64MiB".into());
    }
    let mut writer = guard.writer(Path::new(output), bytes.len() as u64)?;
    writer.write_all(&bytes)?;
    writer.finish()?;
    Ok(())
}

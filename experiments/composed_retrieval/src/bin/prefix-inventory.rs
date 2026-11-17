//! Check the original plan and all annotated synthetic split fingerprints
//! without fitting a model or evaluating an answer.
use composed_retrieval::{Result, data};
use std::path::Path;

fn main() -> Result<()> {
    let inputs = std::env::args_os().skip(1).collect::<Vec<_>>();
    let [inputs] = inputs.as_slice() else {
        return Err("expected ORIGINAL_PLAN_DIRECTORY".into());
    };
    let report = data::inventory(&data::protocol()?, Path::new(inputs))?;
    serde_json::to_writer_pretty(std::io::stdout().lock(), &report)?;
    println!();
    Ok(())
}

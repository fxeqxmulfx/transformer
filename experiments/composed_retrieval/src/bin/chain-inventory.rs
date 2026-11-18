//! Check the original modular-chain source data and both data seeds.
use composed_retrieval::{Result, chain_data::inventory::inventory};
use std::path::Path;

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let [dir] = args.as_slice() else {
        return Err("expected ORIGINAL_CHAIN_SOURCE_DIRECTORY".into());
    };
    serde_json::to_writer_pretty(std::io::stdout().lock(), &inventory(Path::new(dir))?)?;
    println!();
    Ok(())
}

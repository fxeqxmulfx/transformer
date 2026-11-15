//! Development diagnostic: original training split only, no heldout metrics.
use convex_stress_synthetic_programs::{Result, adapter, fit};
use std::{
    fs::OpenOptions,
    io::{BufWriter, Write},
    path::Path,
};
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let [case, output] = args.as_slice() else {
        return Err("usage: train-only CASE FRESH_JSON".into());
    };
    let input = adapter::training_only(case)?;
    let stage = fit::construct(input, fit::settings()?)?;
    let mut file = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(Path::new(output))?,
    );
    serde_json::to_writer(&mut file, &stage)?;
    file.write_all(b"\n")?;
    file.flush()?;
    eprintln!(
        "training-only {}: {} observations, {} compatible programs; no heldout measurement",
        case,
        stage.input.observations.len(),
        stage.discovery.as_ref().map_or(0, |d| d.compatible.len())
    );
    for endpoint in &stage.endpoints {
        eprintln!(
            "{:?}/{:?}: {}",
            endpoint.prior,
            endpoint.policy.mode,
            endpoint.error.clone().unwrap_or_else(|| {
                let t = &endpoint.trained.as_ref().unwrap().training;
                format!(
                    "CE {} -> {}; effective updates {}",
                    t.initial_ce, t.final_ce, t.work.effective_updates
                )
            })
        );
    }
    Ok(())
}

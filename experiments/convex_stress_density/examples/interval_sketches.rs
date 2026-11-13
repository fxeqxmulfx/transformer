//! Extend reproducible task212 fixtures with a declared interval policy. The
//! parent proposals and supplied earlier candidates are fully replayed first.
use convex_stress_density::{
    Result, comparison,
    interval_search::{self, Input, Policy},
    search,
};
use std::{
    fs,
    path::{Path, PathBuf},
};

fn read<T: serde::de::DeserializeOwned>(p: &Path, limit: u64) -> Result<T> {
    if fs::metadata(p)?.len() > limit {
        return Err("fixture input exceeds byte cap".into());
    }
    Ok(serde_json::from_slice(&fs::read(p)?)?)
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: interval_sketches BASE_SKETCH_FIXTURES FRESH_DIRECTORY".into());
    }
    let base = PathBuf::from(&args[0]);
    let dir = PathBuf::from(&args[1]);
    if dir.exists() {
        return Err("interval sketch fixture destination exists".into());
    }
    fs::create_dir_all(&dir)?;
    for name in ["complete", "masked"] {
        let categorical: search::Input =
            read(&base.join(format!("{name}.input.json")), 4 * 1024 * 1024)?;
        let old: comparison::Comparison = read(
            &base.join(format!("{name}.comparison.json")),
            256 * 1024 * 1024,
        )?;
        let models: Vec<_> = old.prior.iter().map(|p| p.model.clone()).collect();
        comparison::replay(&categorical, &models, &old)?;
        let input = Input {
            categorical,
            policy: Policy::default(),
        };
        let out = interval_search::propose(input.clone())?;
        interval_search::replay(&input, &out)?;
        let combined = interval_search::compare(out.clone(), models.clone())?;
        interval_search::replay_comparison(&input, &models, &combined)?;
        fs::write(
            dir.join(format!("{name}.input.json")),
            serde_json::to_vec_pretty(&input)?,
        )?;
        fs::write(
            dir.join(format!("{name}.proposal.json")),
            serde_json::to_vec_pretty(&out)?,
        )?;
        fs::write(
            dir.join(format!("{name}.comparison.json")),
            serde_json::to_vec_pretty(&combined)?,
        )?;
        println!(
            "{}",
            serde_json::json!({"fixture":name,"selected_without_prior":out.selected,
            "selected_with_prior":combined.selected,"exact_cost":out.exact.cost,"interval_cost":out.cost,
            "attempts":out.attempts.iter().map(|a|serde_json::json!({"exact_attempt":a.exact_attempt,
                "variant":a.variant,"reason":a.reason,"ce":a.train_ce,"iterations":a.iterations_charged,
                "last":a.extraction.as_ref().and_then(|e|e.outcome.trace.last())})).collect::<Vec<_>>()})
        );
    }
    Ok(())
}

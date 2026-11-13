//! Extend original complete/masked interval fixtures, with and without supplied
//! earlier models. No hidden labels are provided to the refiner or selection.
use convex_stress_density::{
    Model, Result, interval_search,
    refinement_search::{self, Policy},
};
use std::{
    fs,
    path::{Path, PathBuf},
};

fn read<T: serde::de::DeserializeOwned>(path: &Path, limit: u64) -> Result<T> {
    if fs::metadata(path)?.len() > limit {
        return Err("refinement input exceeds byte cap".into());
    }
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 3 || !["run", "replay"].contains(&args[0].as_str()) {
        return Err(
            "usage: refined_sketches run|replay INTERVAL_FIXTURES REFINEMENT_DIRECTORY".into(),
        );
    }
    let base = PathBuf::from(&args[1]);
    let dir = PathBuf::from(&args[2]);
    if args[0] == "run" {
        if dir.exists() {
            return Err("refinement destination exists".into());
        }
        fs::create_dir_all(&dir)?;
    }
    for name in ["complete", "masked"] {
        let input: interval_search::Input =
            read(&base.join(format!("{name}.input.json")), 4 * 1024 * 1024)?;
        let old: interval_search::Comparison = read(
            &base.join(format!("{name}.comparison.json")),
            256 * 1024 * 1024,
        )?;
        let models: Vec<Model> = old.earlier.prior.iter().map(|p| p.model.clone()).collect();
        interval_search::replay_comparison(&input, &models, &old)?;
        for retain in [false, true] {
            let prior = if retain { models.clone() } else { vec![] };
            let parent = if retain {
                old.clone()
            } else {
                interval_search::compare(old.proposal.clone(), vec![])?
            };
            let path = dir.join(format!("{name}-retain-{retain}.json"));
            let out = if args[0] == "run" {
                let out =
                    refinement_search::run(&input, &prior, parent.clone(), Policy::default())?;
                fs::write(&path, serde_json::to_vec_pretty(&out)?)?;
                out
            } else {
                read(&path, 512 * 1024 * 1024)?
            };
            refinement_search::replay(&input, &prior, &parent, &out)?;
            println!(
                "{}",
                serde_json::json!({"fixture":name,"retained_prior":retain,"selected":out.selected,
                "assignments_charged":out.assignments_charged,"endpoint_audits":out.endpoint_audits,
                "attempts":out.attempts.iter().map(|a|serde_json::json!({"source":a.source,"reason":a.reason,
                    "ce":a.train_ce,"charged":a.assignments_charged,
                    "seed_errors":a.outcome.as_ref().map(|v|v.steps[0].state.errors),
                    "best_errors":a.outcome.as_ref().and_then(|v|v.best.map(|i|v.steps[i].state.errors))})).collect::<Vec<_>>()})
            );
        }
    }
    Ok(())
}

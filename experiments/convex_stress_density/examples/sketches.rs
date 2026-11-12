//! Categorical complete/masked fixtures, not original benchmark tasks.
use convex_stress_density::{
    Model, Result, comparison,
    random::Random,
    search::{self, Input, SearchOptions},
    solver::Options,
};
use std::{fs, path::PathBuf};

fn main() -> Result<()> {
    let dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .ok_or("usage: sketches FRESH_DIRECTORY")?,
    );
    if dir.exists() {
        return Err("sketch fixture destination exists".into());
    }
    fs::create_dir_all(&dir)?;
    let n = 11;
    let (mut rows, mut cols) = ((0..n).collect::<Vec<_>>(), (0..n).collect::<Vec<_>>());
    let mut rng = Random::new(381);
    rng.shuffle(&mut rows);
    rng.shuffle(&mut cols);
    let truth = Model::construct(n, rows, cols, vec![1, 5, 1, 7, 2, 2, 8, 5, 3, 1, 6])?;
    for masked in [false, true] {
        let name = if masked { "masked" } else { "complete" };
        let input = Input {
            n,
            vocab: n,
            observations: (0..n * n)
                .filter(|cell| !masked || cell % 17 != 0)
                .map(|cell| [cell, truth.output_labels[cell]])
                .collect(),
            options: SearchOptions {
                code_trials: 2,
                ranks: vec![1, 2, 3],
                seed: 73,
                extraction: Options {
                    iterations: 256,
                    ..Options::default()
                },
                total_iterations: 2048,
                model_audits: 4,
            },
        };
        let proposal = search::propose(input.clone())?;
        search::replay(&input, &proposal)?;
        let samples = search::validate(&input)?;
        let previous = convex_stress_decode::search::propose(
            n,
            n,
            &samples,
            convex_stress_decode::search::Options {
                code_trials: 16,
                power_steps: 64,
                refinement_steps: 24,
                seed: 73,
            },
        )?;
        let comparison =
            comparison::compare(proposal.clone(), previous.model.into_iter().collect())?;
        fs::write(
            dir.join(format!("{name}.input.json")),
            serde_json::to_vec_pretty(&input)?,
        )?;
        fs::write(
            dir.join(format!("{name}.proposal.json")),
            serde_json::to_vec_pretty(&proposal)?,
        )?;
        fs::write(
            dir.join(format!("{name}.comparison.json")),
            serde_json::to_vec_pretty(&comparison)?,
        )?;
        println!(
            "{}",
            serde_json::json!({"fixture":name,"observations":input.observations.len(),
            "new_selected":proposal.selected,"selected_with_prior":comparison.selected,"cost":proposal.cost,
            "attempts":proposal.attempts.iter().map(|a|serde_json::json!({"trial":a.trial,"rank":a.rank,
                "reason":a.reason,"ce":a.train_ce,"charged":a.iterations_charged})).collect::<Vec<_>>()})
        );
    }
    Ok(())
}

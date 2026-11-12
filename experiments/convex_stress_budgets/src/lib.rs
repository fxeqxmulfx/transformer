//! CM9: add fixed intermediate spectral budgets to the unchanged CM5 solver.
//! Model/CE/projections/batches come from convex_stress_schatten at deef98cb.
//! Only vocabulary size and the fixed candidate label determine the budgets.
//! Rank is not constrained, learned, or read from the target data.
pub use convex_stress_schatten::*;

pub fn budgets(label: &str, n: usize) -> Result<(f64, f64)> {
    if !(2..=256).contains(&n) {
        return Err("unsupported vocabulary size".into());
    }
    let v = n as f64;
    match label {
        "full_rank" => Ok((v.sqrt(), v * v.sqrt())),
        "rank_one" => Ok((v, v)),
        "shape_2" | "shape_3" => {
            let k = if label == "shape_2" { 2 } else { 3 };
            Ok((
                (k as f64 * v).sqrt().min(v),
                v * (n.div_ceil(k) as f64).sqrt(),
            ))
        }
        _ => Err("unknown spectral budget".into()),
    }
}

/// Eligibility reads visible training shape, not task names or held-out examples.
pub fn check_layout(task: &data::Task) -> Result<()> {
    let mut seen = std::collections::BTreeSet::new();
    if task.splits["train"].is_empty()
        || task.splits["train"].iter().any(|e| {
            e.tokens.len() != 2
                || e.targets.len() != 2
                || e.targets[0] != -100
                || e.targets[1] < 0
                || e.targets[1] as usize >= task.vocab
                || !e.prompt.is_empty()
                || e.tokens.iter().any(|&x| x < 0 || x as usize >= task.vocab)
                || !seen.insert(e.tokens.clone())
        })
    {
        return Err("unsupported training pair layout".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intermediate_matrix_has_the_lean_spectrum_and_old_budgets_exclude_it() -> Result<()> {
        let h: Vec<_> = [
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., -1., -1., 1., 1., -1., -1.,
        ]
        .into_iter()
        .map(|x| C::new(x, 0.))
        .collect();
        let singular = linalg::svd(&h, 4)?.values;
        let operator = singular[0];
        let nuclear: f64 = singular.iter().sum();
        assert!((operator - 8_f64.sqrt()).abs() < 1e-12);
        assert!((nuclear - 2. * 8_f64.sqrt()).abs() < 1e-12);
        assert!(operator > budgets("full_rank", 4)?.0);
        assert!(nuclear > budgets("rank_one", 4)?.1);
        let (r, b) = budgets("shape_2", 4)?;
        assert!(operator <= r + 1e-12 && nuclear <= b + 1e-12);
        Ok(())
    }

    #[test]
    fn fixed_budget_controls_and_small_vocabularies_remain_valid() -> Result<()> {
        for n in [2, 3, 4, 5, 97, 193, 256] {
            assert_eq!(budgets("rank_one", n)?, (n as f64, n as f64));
            for name in ["rank_one", "full_rank", "shape_2", "shape_3"] {
                let (r, b) = budgets(name, n)?;
                Model::new(n, r, b)?.validate_shape()?;
            }
        }
        assert_eq!(budgets("shape_2", 2)?, budgets("rank_one", 2)?);
        assert_eq!(budgets("shape_3", 3)?, budgets("rank_one", 3)?);
        assert!(budgets("shape_2", 1).is_err());
        assert!(budgets("answer_rule", 97).is_err());
        Ok(())
    }

    #[test]
    fn intermediate_observed_fit_survives_all_training_modes_and_saved_replay() -> Result<()> {
        use solver::{Mode, Solver};
        // Two repeated character columns, with the fifth column unobserved.
        // These literal labels are a fixture, not features of the predictor.
        let labels = [
            [0, 0, 0, 0],
            [1, 1, 2, 2],
            [2, 2, 4, 4],
            [3, 3, 1, 1],
            [4, 4, 3, 3],
        ];
        let samples: Vec<_> = labels
            .iter()
            .enumerate()
            .flat_map(|(a, row)| {
                row.iter().enumerate().map(move |(b, &label)| Sample {
                    cell: a * 5 + b + 1,
                    label,
                })
            })
            .collect();
        let (r, b) = budgets("shape_2", 5)?;
        for mode in [Mode::Full, Mode::Mini, Mode::Forward] {
            let model = Model::new(5, r, b)?;
            let mut solver = Solver::new(&model, samples.clone(), mode, 1)?;
            for step in 0..1024 {
                solver.step(&[step % samples.len()])?;
            }
            let (out, _, _) = solver.snapshot()?;
            out.validate(&mut solver.projector)?;
            let ce: f64 = samples
                .iter()
                .map(|s| loss(&out.probabilities[s.cell * 5..(s.cell + 1) * 5], s.label))
                .sum::<f64>()
                / samples.len() as f64;
            assert!(
                ce - ideal_loss(5) < 1e-8,
                "observed CE excess {}",
                ce - ideal_loss(5)
            );
            let bytes = serde_json::to_vec(solver.diagnostic_state())?;
            let mut restored = Solver::new(&model, samples.clone(), mode, 1)?;
            restored.restore_diagnostic_state(serde_json::from_slice(&bytes)?)?;
            assert_eq!(restored.snapshot()?.0.probabilities, out.probabilities);
        }
        Ok(())
    }
}

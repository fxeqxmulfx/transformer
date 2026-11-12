use super::*;
use solver::Solver;

fn fixture() -> Vec<Sample> {
    // Literal construction is a test target, never a model feature. Rows zero
    // and one agree; the label at row one, column four is withheld.
    (0..5)
        .flat_map(|col| {
            (0..5).filter_map(move |row| {
                if row == 1 && col == 4 {
                    None
                } else {
                    Some(Sample {
                        cell: row + 5 * col,
                        label: (col + if row < 2 { 0 } else { row }) % 5,
                    })
                }
            })
        })
        .collect()
}

#[test]
fn grouping_uses_overlap_consistency_and_preserves_observation_multiplicity() -> Result<()> {
    let samples = fixture();
    let groups = Groups::discover(5, &samples, 4)?;
    assert_eq!(groups.rows, [0, 0, 1, 2, 3]);
    assert_eq!(groups.columns, [0, 1, 2, 3, 4]);
    let mapped = groups.map_samples(&samples)?;
    assert_eq!(mapped.len(), 24);
    assert_eq!(mapped.iter().filter(|s| s.cell == 0).count(), 2);
    assert_eq!(Groups::discover(5, &samples, 5)?, Groups::identity(5));
    // Bijectively rename labels: grouping must depend only on agreement.
    let renamed: Vec<_> = samples
        .iter()
        .map(|s| Sample {
            cell: s.cell,
            label: (s.label + 2) % 5,
        })
        .collect();
    assert_eq!(Groups::discover(5, &renamed, 4)?, groups);
    let mut inconsistent = samples.clone();
    inconsistent[1].label = 4;
    assert!(groups.map_samples(&inconsistent).is_err());
    assert_ne!(
        Groups::discover(5, &inconsistent, 4)?.rows[0],
        Groups::discover(5, &inconsistent, 4)?.rows[1]
    );
    Ok(())
}

#[test]
fn weighted_sum_gradient_matches_finite_differences() -> Result<()> {
    let original = fixture();
    let groups = Groups::discover(5, &original, 4)?;
    let mapped = groups.map_samples(&original)?;
    let (r, b) = budgets("rank_one", &groups)?;
    let model = Model {
        groups,
        table: Table::new(5, r, b)?,
    };
    let solver = Solver::new(&model, mapped.clone(), Mode::Full, 1)?;
    assert_eq!(solver.maximum_multiplicity, 2);
    assert_eq!(solver.step_size, 0.5);
    let mut x = model.table.probabilities.clone();
    x[0] += 0.13;
    let g = solver.full_gradient(&x);
    let objective = |point: &[f64]| -> f64 {
        mapped
            .iter()
            .map(|s| loss(&point[s.cell * 5..(s.cell + 1) * 5], s.label))
            .sum()
    };
    for index in [0, 1, 4, 5, 100, 124] {
        let mut plus = x.clone();
        let mut minus = x.clone();
        plus[index] += 1e-5;
        minus[index] -= 1e-5;
        let numerical = (objective(&plus) - objective(&minus)) / 2e-5;
        assert!((g[index] - numerical).abs() < 1e-8);
    }
    Ok(())
}

#[test]
fn transitive_agreement_cannot_merge_conflicting_observations() -> Result<()> {
    let mut samples = vec![];
    for col in 0..8 {
        samples.push(Sample {
            cell: 1 + 9 * col,
            label: 0,
        });
        samples.push(Sample {
            cell: if col < 4 { 9 * col } else { 2 + 9 * col },
            label: 0,
        });
    }
    samples.push(Sample {
        cell: 9 * 8,
        label: 1,
    });
    samples.push(Sample {
        cell: 2 + 9 * 8,
        label: 2,
    });
    let groups = Groups::discover(9, &samples, 4)?;
    assert_eq!(groups.rows[0], groups.rows[1]);
    assert_ne!(groups.rows[1], groups.rows[2]);
    assert!(
        groups
            .evidence
            .iter()
            .any(|e| e.axis == 0 && e.first == 1 && e.second == 2 && !e.accepted)
    );
    assert_eq!(groups.map_samples(&samples)?.len(), samples.len());
    Ok(())
}

#[test]
fn identity_groups_reproduce_the_original_solver_in_all_modes() -> Result<()> {
    let samples = fixture();
    for mode in [Mode::Full, Mode::Mini, Mode::Forward] {
        let table = Table::new(5, 5., 5.)?;
        let model = Model {
            groups: Groups::identity(5),
            table: table.clone(),
        };
        let mut old = convex_stress_budgets::solver::Solver::new(&table, samples.clone(), mode, 1)?;
        let mut new = Solver::new(&model, samples.clone(), mode, 1)?;
        for step in 0..24 {
            let selected = [step % samples.len(), (step + 1) % samples.len()];
            old.step(&selected)?;
            new.step(&selected)?;
        }
        assert_eq!(new.diagnostic_state(), old.diagnostic_state());
        assert_eq!(
            new.snapshot()?.0.table.probabilities,
            old.snapshot()?.0.probabilities
        );
        assert_eq!(new.full_residual()?, old.full_residual()?);
    }
    Ok(())
}

#[test]
fn observed_sharing_recovers_a_withheld_cell_in_all_modes_and_replays() -> Result<()> {
    use metrics::Predictor;
    let original = fixture();
    let groups = Groups::discover(5, &original, 4)?;
    let mapped = groups.map_samples(&original)?;
    let (r, b) = budgets("rank_one", &groups)?;
    let model = Model {
        groups,
        table: Table::new(5, r, b)?,
    };
    for mode in [Mode::Full, Mode::Mini, Mode::Forward] {
        let mut solver = Solver::new(&model, mapped.clone(), mode, 1)?;
        for step in 0..256 {
            solver.step(&[step % mapped.len()])?;
        }
        let (out, _, _) = solver.snapshot()?;
        let ce: f64 = original
            .iter()
            .map(|s| {
                loss(
                    &out.logits(&[(s.cell / 5) as i64, (s.cell % 5) as i64]),
                    s.label,
                )
            })
            .sum::<f64>()
            / original.len() as f64;
        assert!(
            ce - ideal_loss(5) < 1e-8,
            "CE excess {}",
            ce - ideal_loss(5)
        );
        assert!(out.logits(&[4, 1])[4] > 1. - 1e-8);
        out.table.validate(&mut solver.projector)?;
        let bytes = serde_json::to_vec(&out)?;
        let saved: Model = serde_json::from_slice(&bytes)?;
        assert_eq!(saved.table.probabilities, out.table.probabilities);
        let mut restored = Solver::new(&model, mapped.clone(), mode, 1)?;
        restored.restore_diagnostic_state(solver.diagnostic_state().to_vec())?;
        assert_eq!(
            restored.snapshot()?.0.table.probabilities,
            out.table.probabilities
        );
    }
    Ok(())
}

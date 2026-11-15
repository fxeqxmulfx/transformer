use crate::{
    model::Prior,
    training::{Mode, Policy},
    *,
};

fn fixture() -> Input {
    let mut observations = vec![];
    for (word, answers) in [
        (vec![3, 4, 4], vec![9, 10, 10, 0]),
        (vec![5, 3, 5, 3], vec![10, 10, 10, 10, 0]),
        (vec![3, 4, 5, 5, 3], vec![10, 9, 10, 10, 10, 0]),
    ] {
        let mut tokens = word;
        tokens.push(1);
        for label in answers {
            observations.push(Observation {
                tokens: tokens.clone(),
                label,
            });
            tokens.push(label);
        }
    }
    Input {
        vocabulary: 16,
        context: 32,
        observations,
    }
}
fn policy(mode: Mode) -> Policy {
    Policy {
        mode,
        iterations: if mode == Mode::Forward { 0 } else { 2 },
        batch: 4,
        learning_rate: 8.,
        initial_margin: 4.,
        minimum_margin: 1.,
        radius: 128.,
    }
}

#[test]
fn training_search_and_categorical_control_expose_missing_label_gap() -> Result<()> {
    let input = fixture();
    let coordinates: Vec<_> = (0..16).map(|c| c as f64 / 16.).collect();
    let report = discovery::run(input.clone(), coordinates)?;
    assert_eq!(report.attempts.len(), 6240);
    assert!(report.attempts.iter().any(|a| a.conflict.is_some()));
    assert!(report.geometry_selection.is_some() && report.categorical_selection.is_some());
    assert!(report.scalar_work <= report.search_forecast);
    let query = [3, 3, 3, 4, 1];
    assert!(!input.observations.iter().any(|r| r.label == 11));
    for mode in [Mode::Forward, Mode::Full, Mode::Mini] {
        let with_prior = model::train(&report, Prior::Coordinates, policy(mode))?;
        let categorical = model::train(&report, Prior::Categorical, policy(mode))?;
        assert_eq!(argmax(&with_prior.model.logits(&query)), 11);
        assert!(with_prior.model.feature(&query).1);
        assert_eq!(argmax(&categorical.model.logits(&query)), 0);
        assert!(!categorical.model.feature(&query).1);
        assert!(with_prior.training.final_ce <= with_prior.training.initial_ce + 1e-12);
        let independent = with_prior.model.independent_logits(&query);
        for (a, b) in with_prior.model.logits(&query).iter().zip(independent) {
            assert!((a - b).abs() < 1e-11);
        }
        for row in &input.observations {
            assert_eq!(argmax(&with_prior.model.logits(&row.tokens)), row.label);
        }
    }
    let saved: discovery::Report = serde_json::from_str(&serde_json::to_string(&report)?)?;
    discovery::replay(&saved)?;
    let mut bad = saved;
    bad.attempts[0].observations += 1;
    assert!(discovery::replay(&bad).is_err());
    Ok(())
}

#[test]
fn input_label_and_budget_failures_do_not_become_candidates() -> Result<()> {
    let coordinates: Vec<_> = (0..16).map(|c| c as f64 / 16.).collect();
    let mut bad = fixture();
    bad.observations[0].label = 16;
    assert!(discovery::run(bad, coordinates.clone()).is_err());
    let mut bad = fixture();
    bad.context = 2;
    assert!(discovery::run(bad, coordinates).is_err());
    Ok(())
}

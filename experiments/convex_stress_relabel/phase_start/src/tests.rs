use super::*;
use convex_stress_relabel::{C, coding, ideal_loss, linalg, loss, metrics::Predictor};

// An independently written categorical fixture, shared with the cold producer's
// tests. The constructor receives only the 24 observed cells, never this array.
const LABELS: [usize; 25] = [
    3, 0, 4, 1, 2, 0, 4, 1, 2, 3, 4, 1, 2, 3, 0, 1, 2, 3, 0, 4, 2, 3, 0, 4, 1,
];
fn observed() -> Vec<Sample> {
    LABELS
        .iter()
        .enumerate()
        .filter(|(cell, _)| *cell != 11)
        .map(|(cell, &label)| Sample { cell, label })
        .collect()
}
fn predict(model: &Model, cell: usize) -> usize {
    let logits = model.logits(&[
        (cell / model.table.vocab) as i64,
        (cell % model.table.vocab) as i64,
    ]);
    (0..model.table.vocab)
        .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
        .unwrap()
}

#[test]
fn constructed_candidate_recovers_the_missing_fixture_and_raw_spectra() -> Result<()> {
    let training = observed();
    let coding = coding::discover(5, &training)?.coding.unwrap();
    let proposal = propose(5, &training, &coding)?;
    assert!(proposal.rejection.is_none());
    assert_eq!(proposal.examples_presented, 24);
    let model = proposal.model.unwrap();
    for (cell, &label) in LABELS.iter().enumerate() {
        assert_eq!(predict(&model, cell), label);
        let logits = model.logits(&[(cell / 5) as i64, (cell % 5) as i64]);
        assert!((loss(&logits, label) - ideal_loss(5)).abs() < 1e-14);
    }
    // Independent direct Fourier sums from serialized coordinates, bypassing
    // the phase constructor, graph checker and producer FFT/projections.
    let serialized = serde_json::to_vec(&model)?;
    let saved: Model = serde_json::from_slice(&serialized)?;
    for f in 1..5 {
        let matrix: Vec<C> = saved
            .table
            .probabilities
            .as_chunks::<5>()
            .0
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(c, &p)| C::from_polar(p, std::f64::consts::TAU * (f * c) as f64 / 5.))
                    .sum()
            })
            .collect();
        let singular = linalg::svd(&matrix, 5)?.values;
        assert!((singular[0] - 5.).abs() < 1e-12);
        assert!((singular.iter().sum::<f64>() - 5.).abs() < 1e-12);
        assert!(singular[1..].iter().all(|x| x.abs() < 1e-12));
    }
    assert_eq!(saved.table.probabilities, model.table.probabilities);
    Ok(())
}

#[test]
fn input_and_output_renamings_preserve_the_constructed_predictions() -> Result<()> {
    let original = observed();
    let rows = [2, 4, 1, 0, 3];
    let columns = [3, 1, 4, 2, 0];
    let classes = [2, 4, 1, 0, 3];
    let mut renamed: Vec<_> = original
        .iter()
        .map(|x| Sample {
            cell: rows[x.cell % 5] + 5 * columns[x.cell / 5],
            label: classes[x.label],
        })
        .collect();
    renamed.reverse();
    let coding = coding::discover(5, &renamed)?.coding.unwrap();
    let proposal = propose(5, &renamed, &coding)?;
    let model = proposal.model.unwrap();
    for (cell, &label) in LABELS.iter().enumerate() {
        assert_eq!(
            predict(&model, rows[cell % 5] + 5 * columns[cell / 5]),
            classes[label]
        );
    }
    Ok(())
}

#[test]
fn graph_conditions_reject_without_substituting_a_model() -> Result<()> {
    let disconnected = propose(
        2,
        &[Sample { cell: 0, label: 0 }, Sample { cell: 3, label: 0 }],
        &Coding::identity(2),
    )?;
    assert_eq!(disconnected.rejection, Some(Rejection::Disconnected));
    assert!(disconnected.model.is_none());
    let cross: Vec<_> = [(0, 0), (3, 1), (6, 2), (1, 1), (2, 2)]
        .into_iter()
        .map(|(cell, label)| Sample { cell, label })
        .collect();
    let unmatched = propose(3, &cross, &Coding::identity(3))?;
    assert_eq!(unmatched.rejection, Some(Rejection::NoPerfectMatching));
    assert!(unmatched.model.is_none());
    let conflict: Vec<_> = [0, 0, 0, 1]
        .into_iter()
        .enumerate()
        .map(|(cell, label)| Sample { cell, label })
        .collect();
    let inconsistent = propose(2, &conflict, &Coding::identity(2))?;
    assert_eq!(inconsistent.rejection, Some(Rejection::InconsistentPhases));
    assert!(inconsistent.model.is_none());
    assert!(propose(194, &[], &Coding::identity(194)).is_err());
    assert!(propose(usize::MAX, &[], &Coding { encode: vec![] }).is_err());
    assert!(propose(5, &observed(), &Coding { encode: vec![0; 5] }).is_err());
    let mut duplicate = observed();
    duplicate.push(duplicate[0].clone());
    assert!(propose(5, &duplicate, &Coding::identity(5)).is_err());
    duplicate.push(duplicate[0].clone());
    assert!(propose(5, &duplicate, &Coding::identity(5)).is_err());
    assert!(propose(5, &[], &Coding::identity(5)).is_err());
    Ok(())
}

#[test]
fn a_changed_unseen_answer_does_not_change_training_or_the_candidate() -> Result<()> {
    let training = observed();
    let code = coding::discover(5, &training)?.coding.unwrap();
    let model = propose(5, &training, &code)?.model.unwrap();
    let mut alternative_truth = LABELS;
    alternative_truth[11] = (LABELS[11] + 1) % 5;
    assert!(
        training
            .iter()
            .all(|x| alternative_truth[x.cell] == x.label)
    );
    assert_ne!(predict(&model, 11), alternative_truth[11]);
    assert_eq!(
        model.table.probabilities,
        propose(5, &training, &code)?
            .model
            .unwrap()
            .table
            .probabilities
    );
    Ok(())
}

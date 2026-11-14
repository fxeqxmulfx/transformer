use super::*;
use crate::model::{Encoded, Model};
fn nontrivial() -> Result<(Model, Vec<Encoded>)> {
    Ok((
        Model {
            head: head::Head::new(
                5,
                (0..3)
                    .map(|j| (0..STATES).map(|s| Some((s + j) % 5)).collect())
                    .collect(),
            )?,
            context: 3,
            alphabet: vec![0, 1, 2, 3],
            programs: program::catalogue(&[0, 1, 2, 3], 5)?[..3].to_vec(),
            weights: vec![0.2, 0.3, 0.5],
        },
        (0..11)
            .map(|i| Encoded {
                states: vec![i % 3, (i + 1) % 3, (i + 2) % 3],
                label: (i + 2) % 5,
            })
            .collect(),
    ))
}

#[test]
fn full_vocabulary_derivative_matches_finite_differences_of_actual_products() -> Result<()> {
    let (model, samples) = nontrivial()?;
    model.validate()?;
    model.head.audit(&model.weights)?;
    let table = model.head.independent_table();
    let (loss, g) = training::gradient(&model, &samples, &table)?;
    assert!((loss - training::score(&model, &samples)).abs() < 1e-12);
    assert!(g.iter().any(|v| v.abs() > 0.1));
    for (j, &slope) in g.iter().enumerate() {
        let mut plus = model.clone();
        let mut minus = model.clone();
        plus.weights[j] += 1e-6;
        minus.weights[j] -= 1e-6;
        let finite = (training::score(&plus, &samples) - training::score(&minus, &samples)) / 2e-6;
        assert!((slope - finite).abs() < 2e-7, "{slope} vs {finite}");
    }
    let mut improved = model.clone();
    for (w, d) in improved.weights.iter_mut().zip(g) {
        *w -= 0.01 * d;
    }
    training::project(&mut improved.weights)?;
    assert!(training::score(&improved, &samples) < loss);
    assert_ne!(model.weights, improved.weights);
    Ok(())
}

#[test]
fn weighted_minibatches_and_short_tail_equal_full_loss_and_gradient() -> Result<()> {
    let (model, samples) = nontrivial()?;
    let table = model.head.independent_table();
    let (full, gradient) = training::gradient(&model, &samples, &table)?;
    let mut weighted = 0.;
    let mut g = vec![0.; gradient.len()];
    let sizes: Vec<_> = samples.chunks(4).map(|s| s.len()).collect();
    assert_eq!(sizes, vec![4, 4, 3]);
    for batch in samples.chunks(4) {
        let (loss, part) = training::gradient(&model, batch, &table)?;
        let mass = batch.len() as f64 / samples.len() as f64;
        weighted += mass * loss;
        for (out, d) in g.iter_mut().zip(part) {
            *out += mass * d;
        }
    }
    assert!((weighted - full).abs() < 1e-12);
    for (a, b) in g.iter().zip(gradient) {
        assert!((a - b).abs() < 1e-12);
    }
    let mut projected = vec![-0.7, 2.1, 0.5];
    training::project(&mut projected)?;
    assert_eq!(projected, vec![0., 1., 0.]);
    Ok(())
}

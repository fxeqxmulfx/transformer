use super::*;

fn loss(model: &Model, ids: &[usize]) -> Result<f64> {
    Ok(ids
        .iter()
        .map(|&i| {
            let [cell, label] = model.records[i];
            model
                .logits_cell(cell)
                .map(|z| kernel::training::ce(&z, label))
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .sum::<f64>()
        / ids.len() as f64)
}

#[test]
fn nonzero_derivative_matches_actual_complex_ce_with_repeated_records_and_tail() -> Result<()> {
    let (raw, grouped) = fixture()?;
    let mut prior = kernel::initialize(&raw)?.model;
    let mut supplement = grouped::construct::initialize(&grouped)?.model;
    for (number, model) in [&mut prior, &mut supplement].into_iter().enumerate() {
        for b in &model.blocks {
            for (s, &label) in b.observed_labels.iter().enumerate() {
                for c in 0..model.vocabulary {
                    model.table[(b.offset + s) * model.vocabulary + c] = if c == label {
                        0.7 + number as f64
                    } else {
                        -0.1 * (c + 1) as f64
                    };
                }
            }
        }
        model.materialize();
    }
    let source = &raw.request.input.observations;
    let records = vec![source[0], source[1], source[0], source[2]];
    let model = Model::new(prior, supplement, &raw, &grouped, records)?;
    // Different actual logits at an overlapping supported input still preserve
    // the prior exactly. No equality premise about the two heads is assumed.
    assert_ne!(
        bits(model.prior.logits_cell(1)),
        bits(model.supplement.logits_cell(1))
    );
    assert_eq!(
        bits(model.logits_cell(1)?),
        bits(model.prior.logits_cell(1))
    );
    for ids in [&[0, 1, 2, 3][..], &[0, 1, 2][..], &[3][..]] {
        let gradient = model.batch_gradient(ids)?;
        assert_eq!(gradient.loss.to_bits(), loss(&model, ids)?.to_bits());
        assert!(gradient.prior.iter().map(|x| x * x).sum::<f64>() > 0.01);
        let eps = 1e-5;
        for i in 0..model.prior.coefficients.len() {
            let mut plus = model.clone();
            let mut minus = model.clone();
            plus.prior.coefficients[i] += eps;
            minus.prior.coefficients[i] -= eps;
            let numerical = (loss(&plus, ids)? - loss(&minus, ids)?) / (2. * eps);
            assert!(
                (numerical - gradient.prior[i]).abs() < 2e-9,
                "coordinate{i}"
            );
        }
        let mut changed_supplement = model.clone();
        for value in &mut changed_supplement.supplement.coefficients {
            *value += 0.17;
        }
        assert_eq!(
            loss(&changed_supplement, ids)?.to_bits(),
            loss(&model, ids)?.to_bits()
        );
    }
    assert!((loss(&model, &[0, 1, 2, 3])? - loss(&model, &[0, 1, 3])?).abs() > 1e-4);
    for ids in [vec![], vec![0, 0], vec![4], vec![0, 1, 2, 3, 4]] {
        assert!(model.batch_gradient(&ids).is_err());
    }
    Ok(())
}

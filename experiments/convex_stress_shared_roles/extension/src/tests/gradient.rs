use super::*;
fn loss(model: &Model, positions: &[usize]) -> Result<f64> {
    Ok(positions
        .iter()
        .map(|&i| {
            let record = &model.records[i];
            Ok(shared::training::ce(
                &model.logits_cell(record.cell)?,
                record.label,
            ))
        })
        .collect::<Result<Vec<_>>>()?
        .iter()
        .sum::<f64>()
        / positions.len() as f64)
}
#[test]
fn actual_nonzero_complex_derivatives_keep_repeated_records_and_final_tail() -> Result<()> {
    let (raw, shared) = fixture()?;
    let mut prior = kernel::initialize(&raw)?.model;
    let mut supplement = shared::construct::initialize(&shared)?.model;
    for (index, model) in [&mut prior, &mut supplement].into_iter().enumerate() {
        for b in &model.blocks {
            for s in 0..b.observed.len() {
                for c in 0..model.vocabulary {
                    model.table[(b.offset + s) * model.vocabulary + c] =
                        (1 + s + 2 * c) as f64 * (index + 1) as f64 / 10.;
                }
            }
        }
        model.materialize();
        model.validate()?;
    }
    let mut records = samples(&raw.request.input);
    let mut repeated = records[0].clone();
    repeated.id = 99;
    records.insert(2, repeated);
    let model = Model::new(prior, supplement, &raw, &shared, records)?;
    assert_ne!(
        bits(model.prior.logits_cell(model.records[0].cell)),
        bits(model.supplement.logits_cell(model.records[0].cell))
    );
    let epsilon = 1e-6;
    for positions in [&[0, 1, 2, 3][..], &[0, 1, 2][..], &[3][..]] {
        let gradient = model.batch_gradient(positions)?;
        let records = positions
            .iter()
            .map(|&i| model.records[i].clone())
            .collect::<Vec<_>>();
        let reference =
            shared::training::gradient(&model.prior, &records::encode(&model.prior, &records)?)?;
        assert_eq!(gradient.loss.to_bits(), reference.loss.to_bits());
        assert_eq!(bits(gradient.prior.clone()), bits(reference.head));
        assert_eq!(gradient.loss.to_bits(), loss(&model, positions)?.to_bits());
        assert!(gradient.prior.iter().any(|x| x.abs() > 1e-3));
        assert!(gradient.supplement.iter().all(|&x| x == 0.));
        for i in 0..model.prior.coefficients.len() {
            let mut plus = model.clone();
            let mut minus = model.clone();
            plus.prior.coefficients[i] += epsilon;
            minus.prior.coefficients[i] -= epsilon;
            let numerical = (loss(&plus, positions)? - loss(&minus, positions)?) / (2. * epsilon);
            assert!(
                (numerical - gradient.prior[i]).abs() < 2e-7,
                "prior coordinate {i}"
            );
        }
        for i in 0..model.supplement.coefficients.len() {
            let mut plus = model.clone();
            let mut minus = model.clone();
            plus.supplement.coefficients[i] += epsilon;
            minus.supplement.coefficients[i] -= epsilon;
            assert_eq!(
                loss(&plus, positions)?.to_bits(),
                loss(&minus, positions)?.to_bits()
            );
        }
    }
    assert!(model.batch_gradient(&[]).is_err());
    assert!(model.batch_gradient(&[0, 0]).is_err());
    assert!(model.batch_gradient(&[4]).is_err());
    Ok(())
}

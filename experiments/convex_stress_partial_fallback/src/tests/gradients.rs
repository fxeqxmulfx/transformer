use super::*;
#[test]
fn repeated_raw_records_and_all_complex_gradient_coordinates_remain_exact() -> Result<()> {
    let mut x = request();
    x.input.observations.push(x.input.observations[0].clone());
    let mut head = candidate(&x)?;
    for w in &mut head.coefficients {
        *w *= 0.02;
    }
    let model = Model::new(head.clone(), parent(7, 5)?, &x.input)?;
    let encoded = head.observations(&x.input)?;
    let saved_parent = serde_json::to_vec(model.parent())?;
    for ids in [vec![0, 1, 2, 3, 4], vec![0, 1], vec![2, 3], vec![4]] {
        let (loss, gradient) = model.batch_gradient(&x.input, &ids)?;
        let reference = causal::training::gradient(
            &head,
            &ids.iter().map(|&i| encoded[i]).collect::<Vec<_>>(),
        )?;
        assert_eq!((loss, &gradient), (reference.0, &reference.1));
        assert!((loss - batch_loss(&model, &x.input, &ids)?).abs() < 1e-14);
        for (index, derivative) in gradient.iter().enumerate() {
            let epsilon = 1e-5;
            let mut plus = head.clone();
            let mut minus = head.clone();
            plus.coefficients[index] += epsilon;
            minus.coefficients[index] -= epsilon;
            let p = Model::new(plus, model.parent().clone(), &x.input)?;
            let m = Model::new(minus, model.parent().clone(), &x.input)?;
            let actual = (batch_loss(&p, &x.input, &ids)? - batch_loss(&m, &x.input, &ids)?)
                / (2. * epsilon);
            assert!((actual - derivative).abs() < 2e-9, "coordinate {index}");
        }
    }
    assert_eq!(serde_json::to_vec(model.parent())?, saved_parent);
    for ids in [vec![], vec![0, 0], vec![5]] {
        assert!(model.batch_gradient(&x.input, &ids).is_err());
    }
    let mut reordered = x.input;
    reordered.observations.swap(0, 1);
    assert!(model.batch_gradient(&reordered, &[0]).is_err());
    Ok(())
}

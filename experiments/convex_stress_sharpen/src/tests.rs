use super::*;

fn fixture(radius: f64, nuclear: f64, mass: f64) -> Result<(Model, Vec<Sample>)> {
    let mut model = Model {
        groups: Groups::identity(3),
        table: Table::new(3, radius, nuclear)?,
    };
    let mut samples = vec![];
    for col in 0..3 {
        for row in 0..3 {
            // Test target only; the endpoint algorithm never evaluates this rule.
            let label = (row + col) % 3;
            let cell = row + 3 * col;
            for c in 0..3 {
                model.table.probabilities[cell * 3 + c] =
                    (1. - mass) / 3. + mass * f64::from(c == label);
            }
            samples.push(Sample { cell, label });
        }
    }
    Ok((model, samples))
}

#[test]
fn post_decode_norm_check_repairs_an_infeasible_vertex_without_changing_predictions() -> Result<()>
{
    let (model, samples) = fixture(3_f64.sqrt(), 3. * 3_f64.sqrt(), 0.3)?;
    assert!(
        decoded(&model)?
            .table
            .validate(&mut Projector::new(3, 1)?)
            .is_err()
    );
    let outcome = polish(&model, &samples, 1, 1e-8, 1e-12)?;
    assert!(outcome.certificate.mixture_alpha < 0.6);
    assert_eq!(outcome.certificate.selected, "sharpened");
    assert!(same_decisions(&model, &outcome.selected)?);
    outcome
        .selected
        .table
        .validate(&mut Projector::new(3, 1)?)?;
    assert!(
        outcome.certificate.proposed_training_ce - ideal_loss(3)
            <= outcome.certificate.proposed_excess_bound.unwrap() + 1e-12
    );
    Ok(())
}

#[test]
fn sharpening_can_close_the_ce_gap_but_retains_already_ideal_coordinates() -> Result<()> {
    let (model, samples) = fixture(3., 3., 0.8)?;
    let outcome = polish(&model, &samples, 1, 1e-8, 1e-12)?;
    assert_eq!(outcome.certificate.selected, "sharpened");
    assert!(outcome.certificate.selected_training_ce - ideal_loss(3) < 1e-8);
    let (ideal, samples) = fixture(3., 3., 1.)?;
    let unchanged = polish(&ideal, &samples, 1, 1e-8, 1e-12)?;
    assert_eq!(unchanged.certificate.reason, "already_ideal");
    assert_eq!(
        unchanged.selected.table.probabilities,
        ideal.table.probabilities
    );
    Ok(())
}

#[test]
fn wrong_observed_decisions_are_not_promoted_and_duplicate_samples_keep_their_weight() -> Result<()>
{
    let (model, mut samples) = fixture(3., 3., 0.8)?;
    samples[0].label = 1;
    let single = loss(&model.table.probabilities[..3], 1);
    let before = training_ce(&model, &samples)?;
    samples.push(samples[0].clone());
    assert!((training_ce(&model, &samples)? - (9. * before + single) / 10.).abs() < 1e-12);
    let outcome = polish(&model, &samples, 1, 1e-8, 1e-12)?;
    assert_eq!(outcome.certificate.reason, "observed_decoding_error");
    assert!(outcome.certificate.proposed_excess_bound.is_none());
    assert_eq!(
        outcome.selected.table.probabilities,
        model.table.probabilities
    );
    Ok(())
}

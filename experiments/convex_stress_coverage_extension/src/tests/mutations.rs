use super::*;

#[test]
fn coefficients_support_records_and_accounting_cannot_change_on_load() -> Result<()> {
    let (model, raw, grouped) = initialized()?;
    for kind in 0..8 {
        let mut value = serde_json::to_value(&model)?;
        match kind {
            0 => value["prior"]["coefficients"][4] = serde_json::json!(0.5),
            1 => value["supplement"]["coefficients"][4] = serde_json::json!(0.5),
            2 => value["prior"]["encoding"]["cell_blocks"][4] = serde_json::json!(0),
            3 => value["records"][0][1] = serde_json::json!(1),
            4 => value["bounds"]["radius"] = serde_json::json!(0.),
            5 => value["coverage"]["newly_covered"] = serde_json::json!(0),
            6 => value["prior"]["blocks"][0]["observed_labels"][0] = serde_json::json!(1),
            _ => value["unexpected"] = serde_json::json!(true),
        }
        assert!(
            Model::load(value, &raw, &grouped, model.binding()).is_err(),
            "mutation{kind}"
        );
    }
    let mut changed = model.prior.clone();
    changed.coefficients[4] += 0.1;
    // Independent numerical checks reject a forged coefficient tensor even
    // when there is no old external hash to compare against.
    assert!(
        Model::new(
            changed,
            model.supplement.clone(),
            &raw,
            &grouped,
            model.records.clone()
        )
        .is_err()
    );
    let mut bad = raw.clone();
    bad.field
        .as_mut()
        .unwrap()
        .space
        .as_mut()
        .unwrap()
        .directions
        .pop();
    assert!(
        Model::new(
            model.prior.clone(),
            model.supplement.clone(),
            &bad,
            &grouped,
            model.records.clone()
        )
        .is_err()
    );
    let mut bad = grouped.clone();
    bad.request.readout[0] = 1;
    assert!(
        Model::new(
            model.prior.clone(),
            model.supplement.clone(),
            &raw,
            &bad,
            model.records.clone()
        )
        .is_err()
    );
    assert!(
        Model::new(
            model.prior.clone(),
            model.supplement.clone(),
            &raw,
            &grouped,
            model.records[..2].to_vec()
        )
        .is_err()
    );
    let mut swapped = model.records.clone();
    swapped.swap(0, 1);
    let changed_order = Model::new(
        model.prior.clone(),
        model.supplement.clone(),
        &raw,
        &grouped,
        swapped,
    )?;
    assert_ne!(changed_order.binding(), model.binding());
    assert!(
        Model::load(
            serde_json::to_value(changed_order)?,
            &raw,
            &grouped,
            model.binding()
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn both_head_forecasts_and_overflows_are_charged_before_audit() -> Result<()> {
    let (model, raw, grouped) = initialized()?;
    let mut request = grouped.request.clone();
    request.input.policy.elimination.scalar_updates -= 1;
    let other_policy = grouped::discovery::run(request)?;
    Model::new(
        model.prior.clone(),
        grouped::construct::initialize(&other_policy)?.model,
        &raw,
        &other_policy,
        model.records.clone(),
    )?;
    let a = model.prior.bounds()?;
    let b = model.supplement.bounds()?;
    let cert =
        raw.buffer_preflight_bytes + grouped.grouped.as_ref().unwrap().buffer_preflight_bytes;
    let total = bounds::combine(&a, &b, cert, model.records.len())?;
    assert_eq!(&total, model.bounds());
    assert_eq!(
        total.buffer_bytes,
        a.buffer_bytes + b.buffer_bytes + cert + 128 * model.records.len() as u64
    );
    assert_eq!(total.head_audit_work, a.audit_work + b.audit_work);
    let mut large = a.clone();
    large.buffer_bytes = bounds::BUFFER_CAP;
    assert!(bounds::combine(&large, &b, 0, 1).is_err());
    large.buffer_bytes = u64::MAX;
    assert!(bounds::combine(&large, &b, 0, 1).is_err());
    large = a.clone();
    large.audit_work = bounds::WORK_CAP;
    assert!(bounds::combine(&large, &b, 0, 1).is_err());
    large = a.clone();
    large.real_coefficients = usize::MAX;
    assert!(bounds::combine(&large, &b, 0, 1).is_err());
    assert!(bounds::combine(&a, &b, u64::MAX, 1).is_err());
    assert!(bounds::combine(&a, &b, 0, 0).is_err());
    assert!(bounds::combine(&a, &b, 0, bounds::RECORD_CAP + 1).is_err());
    Ok(())
}

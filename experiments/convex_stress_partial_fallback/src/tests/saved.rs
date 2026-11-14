use super::*;

#[test]
fn routing_version_and_actual_saved_heads_stay_externally_bound() -> Result<()> {
    let x = request();
    let model = Model::new(candidate(&x)?, parent(7, 5)?, &x.input)?;
    let value = serde_json::to_value(&model)?;
    let restored = Model::load(value.clone(), &x.input, model.binding())?;
    assert_eq!(restored.logits(&[0, 2, 6])?, model.logits(&[0, 2, 6])?);
    assert_eq!(restored.logits(&[1; 65])?, model.logits(&[1; 65])?);
    for change in 0..8 {
        let mut bad = value.clone();
        match change {
            0 => bad["routing_policy"] = json!("parent_everywhere"),
            1 => bad["original"]["candidate"]["coefficients"][0] = json!(0.123),
            2 => bad["original"]["parent"]["model"]["scale"] = json!(8.),
            3 => bad["original"]["bounds"]["radius"] = json!(1.),
            4 => bad["original"]["candidate"]["radius"] = json!(0.),
            5 => bad["cached_predictions"] = json!([]),
            6 => bad["original"]["bounds"]["buffer_bytes"] = json!(0),
            _ => {
                bad["original"]["parent"]["model"]["scale"] = json!(8.);
                let p: Parent = serde_json::from_value(bad["original"]["parent"].clone())?;
                bad["original"]["binding"]["parent_sha256"] = json!(original::digest(&p)?);
            }
        }
        assert!(Model::load(bad, &x.input, model.binding()).is_err());
    }
    let mut changed_input = x.input;
    changed_input.observations[0].label = 0;
    assert!(Model::load(value, &changed_input, model.binding()).is_err());
    Ok(())
}

#[test]
fn original_domains_budgets_and_record_caps_still_apply() -> Result<()> {
    let x = request();
    let head = candidate(&x)?;
    let prior = parent(7, 5)?;
    let model = Model::new(head.clone(), prior.clone(), &x.input)?;
    let a = original::parent::causal_bounds(&head)?;
    let b = prior.bounds()?;
    assert_eq!(model.bounds().radius, a.radius + b.radius);
    assert_eq!(model.bounds().real_slots, a.real_slots + b.real_slots);
    assert_eq!(
        model.bounds().stored_scalars,
        a.stored_scalars + b.stored_scalars
    );
    assert_eq!(
        model.bounds().buffer_bytes,
        a.buffer_bytes + b.buffer_bytes + 64 * 12 + 128 * 4
    );
    assert!(model.bounds().buffer_bytes <= 56 * 1024 * 1024 * 1024);
    for prefix in [vec![], vec![-1], vec![7], vec![1; 4097]] {
        assert!(model.route(&prefix).is_err());
        assert!(model.known(&prefix).is_err());
        assert!(model.logits(&prefix).is_err());
    }
    assert!(Model::new(head.clone(), parent(7, 4)?, &x.input).is_err());
    assert!(Model::new(head.clone(), parent(8, 5)?, &x.input).is_err());
    let oversized = Input {
        vocabulary: 7,
        context: 5,
        observations: vec![x.input.observations[0].clone(); 200_001],
    };
    assert!(Model::new(head.clone(), prior.clone(), &oversized).is_err());
    let mut forged = count_value(7, 5);
    forged["features"]["base"]["context"] = json!(4097);
    assert!(Parent::from_saved("counts", forged).is_err());
    let mut extra = x.clone();
    extra.input.observations.push(causal::Observation {
        tokens: vec![0, 0, 5],
        label: 0,
    });
    assert!(Model::new(candidate(&extra)?, prior, &x.input).is_err());
    let p: Value = serde_json::from_str(include_str!("../../protocol.json"))?;
    assert_eq!(p["routing_policy"], ROUTING_POLICY);
    assert_eq!(p["prediction_limit"], model.prediction_limit());
    assert_eq!(p["memory_limit_bytes"], 68719476736_u64);
    assert_eq!(p["combined_buffer_cap_bytes"], 60129542144_u64);
    assert_eq!(p["new_optimizer_updates"], 0);
    assert_eq!(p["unseen_correctness_certified"], false);
    Ok(())
}

use super::*;
use serde_json::json;
#[test]
fn external_bindings_reject_serialized_parameters_certificates_and_record_edits() -> Result<()> {
    let (model, (raw, shared)) = initialized()?;
    let saved = serde_json::to_value(&model)?;
    Model::load(saved.clone(), &raw, &shared, model.binding())?;
    for kind in 0..8 {
        let mut changed = saved.clone();
        match kind {
            0 => changed["prior"]["coefficients"][0] = json!(19.),
            1 => changed["supplement"]["coefficients"][0] = json!(17.),
            2 => changed["coverage"]["newly_covered"] = json!(0),
            3 => changed["records"][0]["id"] = json!(11),
            4 => changed["records"][0]["label"] = json!(0),
            5 => changed["bounds"]["buffer_bytes"] = json!(0),
            6 => changed["binding"]["shared_discovery_sha256"] = json!("0".repeat(64)),
            _ => changed["records"].as_array_mut().unwrap().swap(0, 1),
        }
        assert!(
            Model::load(changed, &raw, &shared, model.binding()).is_err(),
            "mutation {kind}"
        );
    }
    for kind in 0..4 {
        let mut changed = shared.clone();
        match kind {
            0 => changed.alignments[0][0] += 1,
            1 => changed.buffer_preflight_bytes = 0,
            2 => changed.request.readout[0] = 1,
            _ => changed.shared_role_target_proved = true,
        }
        assert!(
            Model::new(
                model.prior.clone(),
                model.supplement.clone(),
                &raw,
                &changed,
                model.records.clone()
            )
            .is_err()
        );
    }
    for kind in 0..3 {
        let mut changed = raw.clone();
        match kind {
            0 => changed.buffer_preflight_bytes = 0,
            1 => changed.code = None,
            _ => changed.request.prime = 5,
        }
        assert!(
            Model::new(
                model.prior.clone(),
                model.supplement.clone(),
                &changed,
                &shared,
                model.records.clone()
            )
            .is_err()
        );
    }
    let mut duplicate = model.records.clone();
    duplicate[1].id = duplicate[0].id;
    assert!(
        Model::new(
            model.prior.clone(),
            model.supplement.clone(),
            &raw,
            &shared,
            duplicate
        )
        .is_err()
    );
    assert!(
        Model::new(
            model.prior.clone(),
            model.supplement.clone(),
            &raw,
            &shared,
            model.records[..2].to_vec()
        )
        .is_err()
    );
    let mut reordered = model.records.clone();
    reordered.swap(0, 1);
    let other = Model::new(
        model.prior.clone(),
        model.supplement.clone(),
        &raw,
        &shared,
        reordered,
    )?;
    assert_ne!(other.binding.records_sha256, model.binding.records_sha256);
    assert!(Model::load(serde_json::to_value(other)?, &raw, &shared, model.binding()).is_err());
    Ok(())
}

#[test]
fn both_heads_actual_augmented_certificate_and_record_buffers_are_charged() -> Result<()> {
    let (model, (raw, shared)) = initialized()?;
    let a = model.prior.bounds()?;
    let b = model.supplement.bounds()?;
    let certificates = raw.buffer_preflight_bytes + shared.buffer_preflight_bytes;
    let combined = bounds::combine(&a, &b, certificates, model.records.len())?;
    assert_eq!(combined, model.bounds);
    assert_eq!(
        combined.buffer_bytes,
        a.buffer_bytes + b.buffer_bytes + certificates + 128 * model.records.len() as u64
    );
    assert_eq!(combined.head_audit_work, a.audit_work + b.audit_work);
    assert!(bounds::combine(&a, &b, u64::MAX, model.records.len()).is_err());
    assert!(bounds::combine(&a, &b, bounds::BUFFER_CAP, model.records.len()).is_err());
    assert!(bounds::combine(&a, &b, 0, 0).is_err());
    let mut oversized = a.clone();
    oversized.audit_work = bounds::WORK_CAP;
    assert!(bounds::combine(&oversized, &b, 0, model.records.len()).is_err());
    let mut different_policy = shared.request.clone();
    different_policy.input.policy.equations -= 1;
    let alternate = shared::discovery::run(different_policy)?;
    Model::new(
        model.prior.clone(),
        shared::construct::initialize(&alternate)?.model,
        &raw,
        &alternate,
        model.records.clone(),
    )?;
    Ok(())
}

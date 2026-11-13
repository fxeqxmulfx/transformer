use super::*;
use sha2::{Digest, Sha256};
fn fingerprint<T: Serialize>(value: &T) -> Result<String> {
    Ok(Sha256::digest(serde_json::to_vec(value)?)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}
fn tensor_fingerprint(values: &[f64]) -> String {
    let mut hash = Sha256::new();
    hash.update((values.len() as u64).to_le_bytes());
    for value in values {
        hash.update(value.to_bits().to_le_bytes());
    }
    hash.finalize().iter().map(|b| format!("{b:02x}")).collect()
}
#[test]
fn changed_partition_labels_basis_vectors_and_costs_fail_even_after_rehash() -> Result<()> {
    let d = report()?;
    let p = training::policies(3, 2)?;
    let bound = binding::run(&d, p.clone())?;
    for kind in 0..11 {
        let mut bad = d.clone();
        match kind {
            0 => bad.request.readout[0] = 1,
            1 => bad.request.input.observations[1][1] = 1,
            2 => {
                bad.grouped
                    .as_mut()
                    .unwrap()
                    .field
                    .as_mut()
                    .unwrap()
                    .space
                    .as_mut()
                    .unwrap()
                    .directions[0][0] = 1
            }
            3 => bad
                .grouped
                .as_mut()
                .unwrap()
                .field
                .as_mut()
                .unwrap()
                .free
                .clear(),
            4 => {
                bad.grouped
                    .as_mut()
                    .unwrap()
                    .basis_audit
                    .as_mut()
                    .unwrap()
                    .complete = false
            }
            5 => bad.raw_code.as_mut().unwrap().classes[0][0] = 1,
            6 => bad.raw_code.as_mut().unwrap().names.classes.reverse(),
            7 => bad.work.observed_vector_checks -= 1,
            8 => bad.work.ce_gradients = 1,
            9 => bad.raw_label_identification = true,
            _ => bad.unknown_target_compatibility_proved = true,
        }
        let mut rehashed = bound.clone();
        rehashed.discovery_sha256 = fingerprint(&bad)?;
        rehashed.input_sha256 = fingerprint(&bad.request.input)?;
        assert!(
            discovery::check(&d.request, &bad).is_err(),
            "discovery {kind}"
        );
        assert!(
            binding::check(&bad, &p, &rehashed).is_err(),
            "binding {kind}"
        );
    }
    Ok(())
}
#[test]
fn real_imaginary_tensor_dictionary_trajectory_and_policy_mutations_fail() -> Result<()> {
    let d = report()?;
    let p = training::policies(3, 2)?;
    let bound = binding::run(&d, p.clone())?;
    for kind in 0..10 {
        let mut bad = bound.clone();
        let t = bad.endpoints[0].training.as_mut().unwrap();
        match kind {
            0 => t.model.coefficients[0] += 0.1,
            1 => t.model.coefficients[t.model.vocabulary] += 0.1,
            2 => t.model.table[0] += 0.1,
            3 => t.model.encoding.code.classes[0][0] = 1,
            4 => t.model.blocks[0].observed_labels[0] = 1,
            5 => t.steps[0].head_gradient_l2 += 0.1,
            6 => t.scalar_work_forecast -= 1,
            7 => t.work.gradient_examples -= 1,
            8 => bad.target_readout_compatibility_proved = true,
            _ => bad.policies[1].batch = 3,
        }
        let t = bad.endpoints[0].training.as_mut().unwrap();
        let last = t.steps.last_mut().unwrap();
        last.table_sha256 = tensor_fingerprint(&t.model.table);
        last.coefficients_sha256 = tensor_fingerprint(&t.model.coefficients);
        let encoded = serde_json::to_vec(&bad)?;
        let restored: binding::Report = serde_json::from_slice(&encoded)?;
        // A valid serialization/hash cannot replace independent state checks.
        assert_ne!(fingerprint(&restored)?, fingerprint(&bound)?);
        assert!(binding::check(&d, &p, &restored).is_err(), "state {kind}");
    }
    Ok(())
}
#[test]
fn a_refusal_cannot_be_changed_into_a_head_or_free_discovery() -> Result<()> {
    let mut request = fixture();
    request.input.policy.elimination.scalar_updates = 1;
    let d = discovery::run(request.clone())?;
    discovery::replay(&request, &d)?;
    assert_eq!(d.reason, "scalar_update_cap");
    let mut bad = d.clone();
    bad.raw_code = report()?.raw_code;
    assert!(discovery::check(&request, &bad).is_err());
    bad = d.clone();
    bad.work.grouped_observations = 0;
    assert!(discovery::check(&request, &bad).is_err());
    Ok(())
}

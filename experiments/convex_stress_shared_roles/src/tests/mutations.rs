//! Saved matrices, codes, states, original identities and resource refusals.
use super::*;
#[test]
fn mutated_certificates_and_partial_bases_are_rejected() -> Result<()> {
    let mut request = diagonal(3);
    request.input.observations = vec![[0, 0], [4, 1], [8, 2]];
    let out = discovery::run(request.clone())?;
    let field = out.field.as_ref().unwrap();
    let mut short = field.space.as_ref().unwrap().directions.clone();
    short.pop();
    assert!(kernel::discovery::check_basis(&discovery::problem(&request)?, field, &short).is_err());
    let mut mutations = vec![];
    let mut bad = out.clone();
    bad.alignments.pop();
    mutations.push(bad);
    let mut bad = out.clone();
    bad.field.as_mut().unwrap().problem.system.rows[0][0] = 0;
    mutations.push(bad);
    let mut bad = out.clone();
    bad.field
        .as_mut()
        .unwrap()
        .space
        .as_mut()
        .unwrap()
        .directions[0][0] = 1;
    mutations.push(bad);
    let mut bad = out.clone();
    bad.raw_code.as_mut().unwrap().classes[0][0] = 1;
    mutations.push(bad);
    let mut bad = out.clone();
    bad.grouped_code.as_mut().unwrap().rows[0][0] = 1;
    mutations.push(bad);
    let mut bad = out.clone();
    bad.buffer_preflight_bytes -= 1;
    mutations.push(bad);
    let mut bad = out.clone();
    bad.work.role_rows = 0;
    mutations.push(bad);
    let mut bad = out.clone();
    bad.shared_role_target_proved = true;
    mutations.push(bad);
    let mut bad = out.clone();
    bad.target_readout_compatibility_proved = true;
    mutations.push(bad);
    for bad in mutations {
        assert!(discovery::check(&request, &bad).is_err());
    }
    Ok(())
}
#[test]
fn record_binding_and_actual_endpoint_mutations_are_rejected() -> Result<()> {
    let request = diagonal(3);
    let d = discovery::run(request.clone())?;
    let mut samples = records(&request);
    samples.push(records::Record {
        id: 2,
        cell: 0,
        label: 0,
    });
    let policies = training::policies(samples.len(), 2)?;
    let out = binding::run(&d, &samples, policies.clone())?;
    let mut mutations = vec![];
    let mut bad = out.clone();
    bad.original_records -= 1;
    mutations.push(bad);
    let mut bad = out.clone();
    bad.endpoints[0]
        .training
        .as_mut()
        .unwrap()
        .model
        .coefficients[0] += 0.1;
    mutations.push(bad);
    let mut bad = out.clone();
    bad.endpoints[1].training.as_mut().unwrap().steps[1].count = 2;
    mutations.push(bad);
    let mut bad = out.clone();
    bad.endpoints[2]
        .training
        .as_mut()
        .unwrap()
        .model
        .encoding
        .classes[0]
        .clear();
    mutations.push(bad);
    let mut bad = out.clone();
    bad.endpoints[0].audit.as_mut().unwrap().record_ce += 0.1;
    mutations.push(bad);
    let mut bad = out.clone();
    bad.buffer_preflight_bytes = Some(0);
    mutations.push(bad);
    let mut bad = out.clone();
    bad.shared_role_target_proved = true;
    mutations.push(bad);
    for bad in mutations {
        assert!(binding::check(&d, &samples, &policies, &bad).is_err());
    }
    let mut reordered = samples.clone();
    reordered.swap(0, 1);
    assert!(binding::check(&d, &reordered, &policies, &out).is_err());
    let mut duplicate_id = samples.clone();
    duplicate_id[2].id = 0;
    assert!(records::validate(&request.input, &duplicate_id).is_err());
    assert!(
        training::run(
            &out.initialization.as_ref().unwrap().model,
            &duplicate_id,
            policies[0].clone()
        )
        .is_err()
    );
    assert!(records::validate(&request.input, &samples[..1]).is_err());
    Ok(())
}
#[test]
fn missing_alignment_and_resource_refusals_remain_explicit() -> Result<()> {
    let mut request = diagonal(3);
    request.input.observations = vec![[3, 0]];
    let out = discovery::run(request.clone())?;
    assert_eq!(out.reason, "no_shared_categorical_identity");
    discovery::replay(&request, &out)?;
    let mut request = diagonal(3);
    request.input.policy.equations = 2;
    let out = discovery::run(request.clone())?;
    assert_eq!(out.reason, "all_role_equation_cap");
    assert!(out.field.is_none());
    discovery::replay(&request, &out)?;
    let mut request = diagonal(3);
    request.input.policy.elimination.scalar_updates = 0;
    let out = discovery::run(request.clone())?;
    assert!(out.raw_code.is_none());
    discovery::replay(&request, &out)?;
    let request = diagonal(3);
    let d = discovery::run(request.clone())?;
    let samples = records(&request);
    let mut policies = training::policies(samples.len(), 2)?;
    policies[1].updates = 4097;
    let out = binding::run(&d, &samples, policies.clone())?;
    assert!(out.endpoints[1].refusal.is_some());
    binding::replay(&d, &samples, &policies, &out)?;
    let mut huge = out.initialization.as_ref().unwrap().bounds.clone();
    huge.buffer_bytes = u64::MAX;
    assert!(construct::buffer_bytes(&d, &huge).is_err());
    huge.buffer_bytes = 56 * 1024 * 1024 * 1024;
    assert!(construct::buffer_bytes(&d, &huge).is_err());
    Ok(())
}

use super::*;
#[test]
fn all_55_primes_keep_cardinality_refusals_before_elimination() -> Result<()> {
    let primes = kernel::catalogue();
    assert_eq!(primes.len(), 55);
    let mut admitted = 0;
    for prime in primes {
        let mut request = fixture();
        request.prime = prime;
        let d = discovery::run(request.clone())?;
        discovery::replay(&request, &d)?;
        assert_eq!(d.work.partition_entries, 4);
        if prime == 2 {
            assert!(d.raw_code.is_some());
            admitted += 1;
        } else {
            assert_eq!(d.reason, "answer_code_cardinality_mismatch");
            assert!(d.grouped.is_none() && d.raw_code.is_none());
            assert_eq!(d.work.grouped_observations, 0);
            let p = training::policies(3, 2)?;
            let b = binding::run(&d, p.clone())?;
            assert!(b.initialization_refusal.is_some());
            assert!(
                b.endpoints
                    .iter()
                    .all(|e| e.refusal.is_some() && e.training.is_none())
            );
            binding::replay(&d, &p, &b)?;
        }
    }
    assert_eq!(admitted, 1);
    Ok(())
}
#[test]
fn malformed_partition_and_original_inputs_are_rejected() {
    for kind in 0..9 {
        let mut request = fixture();
        match kind {
            0 => request.readout.pop().map(|_| ()).unwrap(),
            1 => request.readout[0] = 2,
            2 => request.readout.fill(0),
            3 => request.answers = 0,
            4 => request.input.observations[0][1] = 4,
            5 => request.input.observations[0][0] = 4,
            6 => request.input.observations[1] = request.input.observations[0],
            7 => request.prime = 4,
            _ => request.input.policy.elimination.nullity = 9,
        }
        assert!(discovery::run(request).is_err(), "invalid {kind}");
    }
}
#[test]
fn full_equation_nullity_and_elimination_caps_keep_their_outcomes() -> Result<()> {
    for kind in 0..4 {
        let mut request = fixture();
        let reason = match kind {
            0 => {
                request.input.policy.equations = 2;
                "all_observed_equation_cap"
            }
            1 => {
                request.input.policy.elimination.nullity = 0;
                "nullity_cap"
            }
            2 => {
                request.input.policy.elimination.scalar_updates = 10;
                "scalar_update_cap"
            }
            _ => {
                request.input.policy.elimination.row_operations = 1;
                "row_operation_cap"
            }
        };
        let d = discovery::run(request.clone())?;
        discovery::replay(&request, &d)?;
        assert_eq!(d.reason, reason);
        assert!(d.raw_code.is_none());
        assert_eq!(d.work.grouped_observations, 3);
        let grouped = d.grouped.as_ref().unwrap();
        if kind > 0 {
            assert!(grouped.field.is_some());
        }
        if kind >= 2 {
            assert!(grouped.work.field.scalar_updates > 0);
        }
        let policies = training::policies(3, 2)?;
        let b = binding::run(&d, policies.clone())?;
        binding::replay(&d, &policies, &b)?;
        assert!(b.endpoints.iter().all(|e| e.refusal.is_some()));
    }
    Ok(())
}
#[test]
fn incompatible_observed_groups_are_refused_after_complete_basis_audit() -> Result<()> {
    let mut request = fixture();
    request.input.observations = vec![[0, 0], [1, 0], [2, 0], [3, 2]];
    let d = discovery::run(request.clone())?;
    discovery::replay(&request, &d)?;
    assert_eq!(d.reason, "incompatible_observed_group_codes");
    assert!(d.raw_code.is_none());
    let grouped = d.grouped.as_ref().unwrap();
    assert!(grouped.basis_audit.as_ref().unwrap().complete);
    assert_eq!(grouped.code.as_ref().unwrap().dimension, 0);
    assert_eq!(d.work.group_pair_comparisons, 1);
    assert!(grouped.work.field.scalar_updates > 0);
    let policies = training::policies(4, 3)?;
    let b = binding::run(&d, policies.clone())?;
    binding::replay(&d, &policies, &b)?;
    assert!(b.initialization.is_none());
    Ok(())
}
#[test]
fn invalid_mode_is_retained_without_discarding_two_valid_modes() -> Result<()> {
    let d = report()?;
    let mut policies = training::policies(3, 2)?;
    policies[1].learning_rate = 2.;
    let b = binding::run(&d, policies.clone())?;
    binding::replay(&d, &policies, &b)?;
    assert!(b.endpoints[0].training.is_some());
    assert_eq!(
        b.endpoints[1].refusal.as_deref(),
        Some("invalid refined CE policy")
    );
    assert!(b.endpoints[2].training.is_some());
    Ok(())
}
#[test]
fn head_and_work_forecasts_refuse_before_large_allocations() -> Result<()> {
    let mut request = fixture();
    request.input.n = 23;
    request.input.observations = (0..529).map(|cell| [cell, 0]).collect();
    let d = discovery::run(request.clone())?;
    discovery::replay(&request, &d)?;
    assert!(d.raw_code.is_some());
    let p = training::policies(529, 256)?;
    let b = binding::run(&d, p.clone())?;
    binding::replay(&d, &p, &b)?;
    assert!(b.initialization.is_none());
    assert!(b.initialization_refusal.as_ref().unwrap().contains("caps"));
    assert!(limits::shape(1024, &[512; 1000]).is_err());
    assert!(
        limits::shape(1024, &[1; 60000])
            .unwrap_err()
            .to_string()
            .contains("forecast")
    );
    let mut small = fixture();
    small.input.vocabulary = 1024;
    small.readout = (0..1024).map(|c| usize::from(c == 2 || c == 3)).collect();
    let d = discovery::run(small)?;
    let initial = construct::initialize(&d)?;
    let samples = initial.model.encode(&d.request.input.observations)?;
    let policy = training::Policy {
        mode: training::Mode::Full,
        updates: 4096,
        batch: 2,
        learning_rate: 1.,
    };
    let forecast = preflight::work(&policy, &initial.model, &samples);
    // The actual fixed three-state case stays bounded even with maximum updates.
    assert!(forecast.is_ok());
    assert!(initial.bounds.buffer_bytes <= 56 * 1024 * 1024 * 1024);
    let mut bigger = fixture();
    bigger.input.n = 5;
    bigger.input.vocabulary = 1024;
    bigger.readout = (0..1024).map(|c| usize::from(c == 1)).collect();
    bigger.input.observations = (0..25).map(|cell| [cell, 0]).collect();
    let d = discovery::run(bigger)?;
    let initial = construct::initialize(&d)?;
    let samples = initial.model.encode(&d.request.input.observations)?;
    let error = preflight::work(&policy, &initial.model, &samples).unwrap_err();
    assert!(error.to_string().contains("32B"));
    assert!(
        training::run(&initial.model, &d.request.input.observations, policy)
            .unwrap_err()
            .to_string()
            .contains("32B")
    );
    Ok(())
}

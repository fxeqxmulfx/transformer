use super::*;
fn request() -> binding::Request {
    let (input, _) = tests::fixture();
    let mut training = training::policies();
    training[0].updates = 3;
    training[1].updates = 5;
    training[1].batch = 19;
    binding::Request {
        discovery: Request { input, prime: 7 },
        training,
        prefix_policy: row_program::Policy::default(),
        certificate_policy: certificate::Policy::default(),
        readout: None,
    }
}
#[test]
fn actual_complex_tables_bounded_heads_all_modes_and_original_certificates() -> Result<()> {
    let mut request = request();
    request.training = training::policies();
    let out = binding::run(request.clone())?;
    binding::replay(&request, &out)?;
    let json = serde_json::to_vec(&out)?;
    assert_eq!(serde_json::from_slice::<binding::Report>(&json)?, out);
    let initial = out.initialization.as_ref().unwrap();
    assert_eq!(
        initial.observed_increments,
        request.discovery.input.observations.len()
    );
    assert_eq!(initial.ce_gradients, 0);
    let code = out.discovery.code.as_ref().unwrap();
    for endpoint in &out.endpoints {
        let model = &endpoint.training.model;
        assert_eq!(model.q, 7);
        assert_eq!(model.vocabulary, 29);
        assert_eq!(endpoint.audit.raw_latent_operator.len(), 7);
        assert_eq!(endpoint.audit.raw_latent_nuclear.len(), 7);
        assert_eq!(endpoint.audit.known_cells, 196);
        assert!(endpoint.audit.train_ce < 0.7);
        assert!(endpoint.audit.raw_training_errors > 0);
        assert!(
            endpoint
                .audit
                .head_column_l1
                .iter()
                .all(|&x| x <= model.radius)
        );
        assert!(!endpoint.certificate.certified); // Unobserved axes remain unknown.
        assert!(!endpoint.certificate.raw_label_identification);
        assert_eq!(endpoint.certificate.classes.len(), 14); // More classes than modulus.
        assert_eq!(
            endpoint.certificate.prefix.input.rows.len(),
            request.discovery.input.observations.len()
        );
        let predictions = model.predictions();
        for [cell, label] in tests::fixture().1 {
            assert_eq!(code.label(predictions[cell]), code.label(label));
            assert_eq!(
                endpoint.certificate.queries[cell].decision,
                certificate::Decision::Passed
            );
        }
        for query in &endpoint.certificate.queries {
            if !model.known(query.cell) {
                assert_ne!(query.decision, certificate::Decision::Passed);
            }
        }
        assert_eq!(
            endpoint.training.work.gradient_batches,
            endpoint.training.policy.updates
        );
        training::replay(
            &initial.model,
            &request.discovery.input.observations,
            &endpoint.training,
        )?;
    }
    assert_eq!(out.endpoints[2].training.model, initial.model);
    assert_eq!(out.endpoints[2].training.work.gradient_examples, 0);
    assert_ne!(
        out.endpoints[0].training.model.coefficients,
        initial.model.coefficients
    );
    assert_ne!(
        out.endpoints[1].training.model.coefficients,
        initial.model.coefficients
    );
    assert!(!out.target_readout_compatibility_proved);
    Ok(())
}

#[test]
fn renamed_categories_preserve_actual_forward_logits_and_certified_codes() -> Result<()> {
    let request = request();
    let out = binding::run(request.clone())?;
    let input = &request.discovery.input;
    let row = |i| (3 * i + 2) % input.n;
    let column = |j| (5 * j + 4) % input.n;
    let class = |c| (2 * c + 7) % input.vocabulary;
    let mut renamed = request.clone();
    renamed.discovery.input.observations = input
        .observations
        .iter()
        .rev()
        .map(|&[cell, label]| {
            [
                row(cell % input.n) + input.n * column(cell / input.n),
                class(label),
            ]
        })
        .collect();
    let mapped = binding::run(renamed.clone())?;
    binding::replay(&renamed, &mapped)?;
    let first = &out.endpoints[2].training.model;
    let second = &mapped.endpoints[2].training.model;
    for [cell, target] in tests::fixture().1 {
        let newcell = row(cell % input.n) + input.n * column(cell / input.n);
        for c in 0..input.vocabulary {
            assert!((first.logits(cell)[c] - second.logits(newcell)[class(c)]).abs() < 1e-12);
        }
        for endpoint in &mapped.endpoints {
            assert_eq!(
                endpoint.certificate.queries[newcell].decision,
                certificate::Decision::Passed
            );
            let code = mapped.discovery.code.as_ref().unwrap();
            assert_eq!(
                code.label(endpoint.training.model.predictions()[newcell]),
                code.label(class(target))
            );
        }
    }
    Ok(())
}
#[test]
fn actual_ce_derivative_matches_finite_difference_and_weighted_minibatches() -> Result<()> {
    let request = request();
    let discovery = run(request.discovery.clone())?;
    let mut model = model::initialize(&discovery)?.model;
    // Non-optimal finite state ensures a test of meaningful derivatives.
    for (k, w) in model.coefficients.iter_mut().enumerate() {
        *w = ((k * 17 + 3) % 11) as f64 / 5. - 1.;
    }
    let samples = &request.discovery.input.observations;
    let (loss, g) = training::gradient(&model, samples)?;
    assert!((loss - training::score(&model, samples)).abs() < 1e-12);
    let eps = 1e-5;
    for k in (0..g.len()).step_by(7) {
        let center = model.coefficients[k];
        model.coefficients[k] = center + eps;
        let a = training::score(&model, samples);
        model.coefficients[k] = center - eps;
        let b = training::score(&model, samples);
        model.coefficients[k] = center;
        assert!(((a - b) / (2. * eps) - g[k]).abs() < 1e-8);
    }
    let mut batched = vec![0.; g.len()];
    for batch in samples.chunks(19) {
        let (_, chunk) = training::gradient(&model, batch)?;
        for (sum, part) in batched.iter_mut().zip(chunk) {
            *sum += part * batch.len() as f64 / samples.len() as f64;
        }
    }
    assert!(batched.iter().zip(&g).all(|(a, b)| (a - b).abs() < 1e-14));
    let trained = training::run(&model, samples, request.training[0].clone())?;
    training::check(&model, samples, &trained)?;
    assert!(training::score(&trained.model, samples) < loss - 0.01);
    Ok(())
}
#[test]
fn declared_readout_cannot_change_training_or_establish_target_compatibility() -> Result<()> {
    let mut request = request();
    let original = binding::run(request.clone())?;
    let code = original.discovery.code.as_ref().unwrap();
    request.readout = Some(
        (0..request.discovery.input.vocabulary)
            .map(|c| code.label(c).map_or(999, usize::from))
            .collect(),
    );
    let compatible = binding::run(request.clone())?;
    binding::check(&request, &compatible)?;
    assert_eq!(compatible.candidate_readout_compatible, Some(true));
    assert!(!compatible.target_readout_compatibility_proved);
    assert_eq!(compatible.endpoints, original.endpoints);
    request.readout = Some((0..request.discovery.input.vocabulary).collect());
    let identity = binding::run(request.clone())?;
    binding::check(&request, &identity)?;
    assert_eq!(identity.candidate_readout_compatible, Some(false));
    assert_eq!(identity.endpoints, original.endpoints);
    Ok(())
}
#[test]
fn altered_saved_models_losses_updates_certificates_and_scope_fail_checks() -> Result<()> {
    let request = request();
    let out = binding::run(request.clone())?;
    for slot in 0..15 {
        let mut bad = out.clone();
        match slot {
            0 => bad.initialization.as_mut().unwrap().counts[0] += 1,
            1 => bad.initialization.as_mut().unwrap().ce_gradients = 1,
            2 => bad.endpoints[0].training.model.latent[0] = 1,
            3 => bad.endpoints[0].training.model.coefficients[0] += 0.1,
            4 => bad.endpoints[1].training.steps[0].gradient_l2 += 0.1,
            5 => bad.endpoints[1].training.steps[0].start += 1,
            6 => bad.endpoints[2].training.work.gradient_batches = 1,
            7 => bad.endpoints[0].audit.raw_latent_nuclear[0] += 1.,
            8 => bad.endpoints[0].audit.train_ce += 0.1,
            9 => bad.endpoints[0].certificate.input.observations[0][1] = 0,
            10 => bad.endpoints[0].training.model.known_rows[0] = true,
            11 => bad.raw_label_identification = true,
            12 => bad.target_readout_compatibility_proved = true,
            13 => bad.candidate_readout_compatible = Some(true),
            _ => {
                bad.endpoints.pop();
            }
        }
        assert!(binding::check(&request, &bad).is_err(), "mutation{slot}");
    }
    Ok(())
}
#[test]
fn refusal_budget_and_original_input_bindings_remain_explicit() -> Result<()> {
    let mut request = request();
    request.discovery.input.policy.elimination.scalar_updates = 0;
    let refused = binding::run(request.clone())?;
    binding::replay(&request, &refused)?;
    assert!(refused.initialization.is_none());
    assert!(refused.endpoints.is_empty());
    request.discovery.input.policy = Policy::default();
    request.certificate_policy.queries = 0;
    let capped = binding::run(request.clone())?;
    binding::replay(&request, &capped)?;
    assert!(
        capped
            .endpoints
            .iter()
            .all(|e| e.certificate.reason == "query_count_cap")
    );
    request.training.swap(0, 1);
    assert!(binding::run(request.clone()).is_err());
    request.training.swap(0, 1);
    request.training[2].updates = 1;
    assert!(binding::run(request.clone()).is_err());
    request.training[2].updates = 0;
    request.readout = Some(vec![0]);
    assert!(binding::run(request).is_err());
    Ok(())
}

//! Actual complex CE, original record multiplicity and long-prefix checks.
use super::*;
#[test]
fn actual_modes_replay_repeated_records_and_tail_batches() -> Result<()> {
    let request = diagonal(3);
    let discovery = discovery::run(request.clone())?;
    let mut samples = records(&request);
    samples.push(records::Record {
        id: 2,
        cell: 0,
        label: 0,
    });
    let policies = training::policies(samples.len(), 2)?;
    let out = binding::run(&discovery, &samples, policies.clone())?;
    binding::replay(&discovery, &samples, &policies, &out)?;
    assert_eq!(out.original_records, 3);
    assert!(out.buffer_preflight_bytes.unwrap() < 56 * 1024 * 1024 * 1024);
    for (index, expected_batches) in [1, 2, 0].into_iter().enumerate() {
        let endpoint = &out.endpoints[index];
        let trained = endpoint.training.as_ref().unwrap();
        assert_eq!(trained.work.gradient_batches, expected_batches);
        assert_eq!(
            trained.work.gradient_examples,
            if index == 2 { 0 } else { 3 }
        );
        assert_eq!(endpoint.audit.as_ref().unwrap().original_records, 3);
        assert_eq!(endpoint.audit.as_ref().unwrap().record_errors, 0);
    }
    let mini = out.endpoints[1].training.as_ref().unwrap();
    assert_eq!(
        mini.steps
            .iter()
            .map(|s| (s.start, s.count))
            .collect::<Vec<_>>(),
        vec![(0, 2), (2, 1)]
    );
    assert_eq!(
        out.endpoints[2].training.as_ref().unwrap().model,
        out.initialization.as_ref().unwrap().model
    );
    let head = prediction::CheckedHead::bind(
        &discovery,
        &samples,
        &policies,
        &out,
        training::Mode::Forward,
    )?;
    for length in [2, 65, 4096] {
        let mut prefix = vec![0; length];
        prefix[length - 1] = 1;
        let predicted = head.predict(&prefix)?;
        assert_eq!(predicted.support, prediction::Support::Supported);
        assert!(!predicted.exact_observed);
        assert!(predicted.logits[0] > predicted.logits[1]);
    }
    assert_eq!(
        head.predict(&[2, 1])?.support,
        prediction::Support::UnknownAxis
    );
    assert!(head.predict(&[0, 0])?.exact_observed);
    assert!(head.predict(&[0]).is_err());
    assert!(head.predict(&vec![0; 4097]).is_err());
    assert!(head.predict(&[3, 0, 1]).is_err());
    let two_request = diagonal(2);
    let two = discovery::run(two_request.clone())?;
    let samples = records(&two_request);
    let policies = training::policies(samples.len(), 2)?;
    let out = binding::run(&two, &samples, policies.clone())?;
    let head =
        prediction::CheckedHead::bind(&two, &samples, &policies, &out, training::Mode::Forward)?;
    assert_eq!(
        head.predict(&[0, 1])?.support,
        prediction::Support::UnmatchedActive
    );
    Ok(())
}
#[test]
fn nonzero_complex_gradients_count_every_original_record() -> Result<()> {
    let request = diagonal(3);
    let d = discovery::run(request.clone())?;
    let mut model = construct::initialize(&d)?.model;
    for block in &model.blocks {
        for s in 0..block.observed.len() {
            for c in 0..model.vocabulary {
                model.table[(block.offset + s) * model.vocabulary + c] =
                    (1 + s + 2 * c) as f64 / 10.;
            }
        }
    }
    model.materialize();
    model.validate()?;
    let mut samples = records(&request);
    samples.push(records::Record {
        id: 2,
        cell: 0,
        label: 0,
    });
    let encoded = records::encode(&model, &samples)?;
    let g = training::gradient(&model, &encoded)?;
    assert!(g.head.iter().any(|x| x.abs() > 0.1));
    let delta = 1e-6;
    for k in 0..model.coefficients.len() {
        let (mut left, mut right) = (model.clone(), model.clone());
        left.coefficients[k] -= delta;
        right.coefficients[k] += delta;
        let numeric =
            (training::score(&right, &encoded) - training::score(&left, &encoded)) / (2. * delta);
        assert!(
            (numeric - g.head[k]).abs() < 1e-7,
            "coordinate {k}: {numeric} != {}",
            g.head[k]
        );
    }
    let single: Vec<_> = encoded
        .iter()
        .map(|s| training::gradient(&model, &[*s]).unwrap())
        .collect();
    for k in 0..g.head.len() {
        assert!((g.head[k] - single.iter().map(|g| g.head[k]).sum::<f64>() / 3.).abs() < 1e-12);
    }
    let deduplicated = training::gradient(&model, &encoded[..2])?;
    assert!(
        g.head
            .iter()
            .zip(&deduplicated.head)
            .any(|(a, b)| (a - b).abs() > 0.01)
    );
    for b in &model.blocks {
        for s in 0..b.observed.len() {
            for c in 0..model.vocabulary {
                let k = (b.offset + s) * model.vocabulary + c;
                let (mut left, mut right) = (model.clone(), model.clone());
                left.table[k] -= delta;
                right.table[k] += delta;
                left.materialize();
                right.materialize();
                let numeric = (training::score(&right, &encoded)
                    - training::score(&left, &encoded))
                    / (2. * delta);
                assert!((numeric - g.table[k]).abs() < 1e-7);
            }
        }
    }
    for policy in training::policies(samples.len(), 2)? {
        let out = training::run(&model, &samples, policy)?;
        training::replay(&model, &samples, &out)?;
        audit::run(&out.model, &request.input, &samples)?;
    }
    Ok(())
}

#[test]
fn declared_readout_retains_raw_aliases_and_the_full_raw_vocabulary() -> Result<()> {
    let mut request = diagonal(3);
    request.input.vocabulary = 6;
    request.input.observations = vec![[0, 0], [1, 1], [4, 0]];
    request.readout = vec![0, 0, 1, 1, 2, 2];
    let d = discovery::run(request.clone())?;
    let code = d.raw_code.as_ref().unwrap();
    assert_eq!(code.label(0), code.label(1));
    assert!(
        !d.raw_label_identification
            && !d.target_readout_compatibility_proved
            && !d.shared_role_target_proved
    );
    let samples = records(&request);
    let policies = training::policies(samples.len(), 2)?;
    let out = binding::run(&d, &samples, policies.clone())?;
    binding::replay(&d, &samples, &policies, &out)?;
    let head =
        prediction::CheckedHead::bind(&d, &samples, &policies, &out, training::Mode::Forward)?;
    let observed = head.predict(&[0, 1])?;
    assert_eq!(observed.logits.len(), 6);
    assert!(observed.exact_observed);
    assert!(observed.logits[1] > observed.logits[0]);
    let unseen = head.predict(&[1, 0])?;
    assert_eq!(unseen.support, prediction::Support::Supported);
    assert!(!unseen.exact_observed);
    assert!(unseen.logits[0] > unseen.logits[1]);
    for length in [2, 65, 4096] {
        let mut prefix = vec![0; length];
        prefix[length - 1] = 1;
        assert_eq!(head.predict(&prefix)?, observed);
    }
    assert_eq!(request.readout[0], request.readout[1]);
    let mut changed = request;
    changed.readout[4] = 1;
    assert!(discovery::check(&changed, &d).is_err());
    Ok(())
}

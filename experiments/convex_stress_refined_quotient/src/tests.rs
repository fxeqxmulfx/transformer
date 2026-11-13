use crate::*;
use std::collections::BTreeMap;
/// Original categorical alias fixture from convex_stress_quotient at49105730.
pub(crate) fn fixture() -> Result<(quotient::Report, Vec<[usize; 2]>, Vec<usize>)> {
    let (n, q, v) = (17, 7, 29);
    let mut targets = vec![];
    let mut observations = vec![];
    let mut readout: Vec<_> = (q..q + v).collect();
    for j in 0..2 * q {
        for i in 0..2 * q {
            let row = 1 + (5 * i) % (2 * q);
            let column = 1 + (3 * j) % (2 * q);
            let code = (i + 3 * j + 1) % q;
            let alias = usize::from((131 * i + 17 * j + 7 * i * j) % 5 < 2);
            let label = 5 + (5 * (2 * code + alias) + 3) % (2 * q);
            let sample = [row + n * column, label];
            readout[label] = code;
            targets.push(sample);
            if (i + 3 * j) % 5 != 0 {
                observations.push(sample);
            }
        }
    }
    let report = quotient::run(quotient::Request {
        prime: q as u16,
        input: quotient::Input {
            n,
            vocabulary: v,
            observations,
            policy: quotient::Policy::default(),
        },
    })?;
    assert!(report.code.is_some());
    Ok((report, targets, readout))
}
fn pooled(model: &Model, cell: usize, readout: &[usize]) -> usize {
    let logits = model.logits_cell(cell);
    let shift = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut groups = BTreeMap::<usize, f64>::new();
    for (c, &z) in logits.iter().enumerate() {
        *groups.entry(readout[c]).or_default() += (z - shift).exp();
    }
    *groups.iter().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0
}
#[test]
fn observed_raw_collisions_are_fixed_and_every_declared_quotient_is_retained() -> Result<()> {
    let (discovery, targets, readout) = fixture()?;
    let prior = quotient::model::initialize(&discovery)?;
    assert!(quotient::training::score(&prior.model, &discovery.request.input.observations) > 0.1);
    let out = initialize(&discovery)?;
    construct::check(&discovery, &out)?;
    let audit = out.model.audit(&discovery.request.input.observations)?;
    assert_eq!(audit.train_errors, 0);
    assert!(audit.train_ce < 1e-6);
    assert_eq!(out.bounds.states, out.observed_states + out.default_states);
    assert!(out.bounds.phase_entries < out.bounds.states.pow(2));
    assert!(audit.singular_min.iter().all(|s| (s - 1.).abs() < 1e-10));
    assert!(
        out.model
            .coefficients
            .iter()
            .enumerate()
            .any(|(i, x)| (i / out.model.vocabulary) % 2 == 1 && x.abs() > 0.1)
    );
    for &[cell, label] in &targets {
        assert_eq!(pooled(&out.model, cell, &readout), readout[label]);
    }
    // At least one unobserved raw alias remains unidentified despite correct pooling.
    assert!(targets.iter().any(|&[cell, label]| {
        let (b, s) = out.model.location(cell);
        s + 1 == out.model.blocks[b].states() && out.model.blocks[b].default_label != label
    }));
    assert!(!out.model.known(0));
    Ok(())
}
#[test]
fn all_modes_replay_with_one_epoch_tail_and_unchanged_unseen_logits() -> Result<()> {
    let (discovery, targets, _) = fixture()?;
    let count = discovery.request.input.observations.len();
    let policies = training::policies(count, 64)?;
    let out = binding::run(&discovery, policies.clone())?;
    binding::replay(&discovery, &policies, &out)?;
    for endpoint in &out.endpoints {
        let t = endpoint.training.as_ref().unwrap();
        assert_eq!(endpoint.audit.as_ref().unwrap().train_errors, 0);
        for &[cell, _] in &targets {
            let (b, s) = t.model.location(cell);
            if s + 1 == t.model.blocks[b].states() {
                for (a, b) in t
                    .model
                    .logits_cell(cell)
                    .iter()
                    .zip(out.initialization.model.logits_cell(cell))
                {
                    assert!((a - b).abs() < 1e-9);
                }
            }
        }
    }
    let mini = out.endpoints[1].training.as_ref().unwrap();
    assert_eq!(mini.work.gradient_examples, count);
    assert_eq!(mini.steps.len(), count.div_ceil(64));
    assert_eq!(mini.steps.last().unwrap().count, count % 64);
    let forward = out.endpoints[2].training.as_ref().unwrap();
    assert!(forward.steps.is_empty());
    assert_eq!(forward.work.gradient_batches, 0);
    assert_eq!(forward.work.head_derivative_terms, 0);
    assert_eq!(forward.model, out.initialization.model);
    Ok(())
}
#[test]
fn clipping_outside_the_box_preserves_actual_defaults_and_budget() -> Result<()> {
    let (discovery, _, _) = fixture()?;
    let mut model = initialize(&discovery)?.model;
    for (i, t) in model.table.iter_mut().enumerate() {
        *t = if i % 2 == 0 { 1000. } else { -1000. };
    }
    model.project();
    model.validate()?;
    let audit = model.audit(&discovery.request.input.observations)?;
    assert!(audit.column_l1.iter().all(|&v| v < model.radius));
    assert!(model.table.contains(&32.) && model.table.contains(&-32.));
    assert!(audit.table_logit_error < 1e-8);
    Ok(())
}
#[test]
fn unused_codes_and_unknown_axes_remain_explicit_defaults() -> Result<()> {
    let d = quotient::run(quotient::Request {
        prime: 7,
        input: quotient::Input {
            n: 3,
            vocabulary: 5,
            observations: vec![[0, 0], [1, 1], [3, 1], [4, 2]],
            policy: quotient::Policy::default(),
        },
    })?;
    let out = initialize(&d)?;
    assert_eq!(out.unsupported_defaults, 4);
    assert!(!out.model.known(8));
    assert_eq!(out.bounds.states, 11);
    construct::check(&d, &out)?;
    Ok(())
}
#[test]
fn category_renaming_and_observation_order_do_not_supply_answer_arithmetic() -> Result<()> {
    let (d, targets, readout) = fixture()?;
    let old = initialize(&d)?;
    let mut request = d.request.clone();
    let rename = |c: usize| (2 * c + 5) % 29;
    for s in &mut request.input.observations {
        s[1] = rename(s[1]);
    }
    request.input.observations.reverse();
    let renamed = quotient::run(request)?;
    let new = initialize(&renamed)?;
    let mut q = vec![0; 29];
    for (c, &group) in readout.iter().enumerate() {
        q[rename(c)] = group;
    }
    for &[cell, label] in &targets {
        assert_eq!(pooled(&old.model, cell, &readout), readout[label]);
        assert_eq!(pooled(&new.model, cell, &q), readout[label]);
    }
    Ok(())
}
#[test]
fn altered_models_initializers_and_claims_fail_independent_checks() -> Result<()> {
    let (d, _, _) = fixture()?;
    let initial = initialize(&d)?;
    for kind in 0..6 {
        let mut bad = initial.clone();
        match kind {
            0 => bad.model.coefficients[0] += 0.25,
            1 => bad.model.blocks[0].code[0] += C::new(0., 0.1),
            2 => bad.model.table[0] += 0.25,
            3 => bad.model.blocks[0].default_label = (bad.model.blocks[0].default_label + 1) % 29,
            4 => bad.observed_states -= 1,
            _ => bad.ce_gradients = 1,
        }
        assert!(construct::check(&d, &bad).is_err());
    }
    let policies = training::policies(d.request.input.observations.len(), 64)?;
    let out = binding::run(&d, policies.clone())?;
    for kind in 0..6 {
        let mut bad = out.clone();
        match kind {
            0 => bad.raw_label_identification = true,
            1 => bad.target_readout_compatibility_proved = true,
            2 => bad.discovery_sha256.push('0'),
            3 => bad.endpoints[2].audit.as_mut().unwrap().train_ce += 1.,
            4 => bad.endpoints.swap(0, 2),
            _ => bad.new_discovery_work = 1,
        }
        assert!(binding::check(&d, &policies, &bad).is_err());
    }
    Ok(())
}
#[test]
fn dimension_work_and_mode_refusals_are_bounded_and_retained() -> Result<()> {
    assert!(limits::shape(1025, &[2, 2]).is_err());
    assert!(limits::shape(29, &[514, 1]).is_err());
    assert!(limits::shape(1024, &[513; 257]).is_err());
    assert!(limits::dimensions(usize::MAX, 7, 29).is_err());
    let (d, _, _) = fixture()?;
    let mut policies = training::policies(d.request.input.observations.len(), 64)?;
    policies[0].updates = 4097;
    let out = binding::run(&d, policies.clone())?;
    assert!(out.endpoints[0].refusal.is_some());
    assert!(out.endpoints[1].training.is_some() && out.endpoints[2].training.is_some());
    binding::check(&d, &policies, &out)?;
    let mut reordered = policies.clone();
    reordered.swap(0, 1);
    assert!(binding::run(&d, reordered).is_err());
    Ok(())
}

#[test]
fn scalar_work_caps_refuse_before_updates_and_preserve_other_modes() -> Result<()> {
    // This shape fits every storage cap but exceeds the audit-work allowance.
    let failure = limits::shape(2, &[400; 8]).unwrap_err().to_string();
    assert!(failure.contains("audit-work forecast"));
    let (mut d, _, _) = fixture()?;
    d.request.input.vocabulary = 256;
    let d = quotient::run(d.request)?;
    let mut policies = training::policies(d.request.input.observations.len(), 64)?;
    policies[0].updates = 4096;
    let out = binding::run(&d, policies.clone())?;
    assert_eq!(
        out.endpoints[0].refusal.as_deref(),
        Some("refined CE training exceeds32B scalar-work forecast")
    );
    assert!(out.endpoints[0].training.is_none() && out.endpoints[0].audit.is_none());
    for endpoint in &out.endpoints[1..] {
        assert!(endpoint.refusal.is_none());
        assert_eq!(endpoint.audit.as_ref().unwrap().train_errors, 0);
    }
    binding::replay(&d, &policies, &out)?;
    Ok(())
}

#[test]
fn saved_json_preserves_actual_complex_states_and_every_update_hash() -> Result<()> {
    let (d, _, _) = fixture()?;
    let policies = training::policies(d.request.input.observations.len(), 64)?;
    let out = binding::run(&d, policies.clone())?;
    let bytes = serde_json::to_vec(&out)?;
    let restored: binding::Report = serde_json::from_slice(&bytes)?;
    assert_eq!(restored, out);
    binding::replay(&d, &policies, &restored)?;
    Ok(())
}

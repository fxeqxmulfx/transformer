use super::*;

#[test]
fn valid_work_cap_keeps_mini_and_forward_and_larger_safe_heads_are_allowed() -> Result<()> {
    // This actual-range shape exceeds the old16M coefficient cap but fits the
    // new prospective memory and scalar-work budgets. No tensor is allocated.
    assert!(limits::shape(256, &vec![1; 36481]).is_ok());
    assert!(limits::shape(1024, &vec![1; 66049]).is_err());
    assert_eq!(
        limits::shape(1024, &vec![1; 60000])
            .unwrap_err()
            .to_string(),
        "kernel head exceeds memory or audit-work forecast"
    );
    let source = fixture()?.request.input.observations;
    let samples = source
        .into_iter()
        .map(|[cell, label]| [cell % 3 + 257 * (cell / 3), label])
        .collect();
    let d = discovery::run(request(257, 7, 2, samples))?;
    let mut p = training::policies(5, 2)?;
    p[0].updates = 4096;
    let out = binding::run(&d, p.clone())?;
    assert_eq!(
        out.endpoints[0].refusal.as_deref(),
        Some("refined CE training exceeds32B scalar-work forecast")
    );
    assert!(out.endpoints[0].training.is_none());
    assert!(out.endpoints[1].training.is_some() && out.endpoints[2].training.is_some());
    binding::check(&d, &p, &out)?;
    Ok(())
}

fn pooled(model: &Model, cell: usize, readout: &[usize]) -> usize {
    let logits = model.logits_cell(cell);
    let shift = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut masses = std::collections::BTreeMap::<usize, f64>::new();
    for (c, &z) in logits.iter().enumerate() {
        *masses.entry(readout[c]).or_default() += (z - shift).exp();
    }
    *masses.iter().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0
}
#[test]
fn multidimensional_raw_aliases_fit_observations_with_real_complex_coefficients() -> Result<()> {
    let d = fixture()?;
    let code = d.code.as_ref().unwrap();
    assert_eq!(code.dimension, 2);
    assert_eq!(code.label(0), code.label(2));
    assert_ne!(code.label(0), code.label(1));
    let out = initialize(&d)?;
    construct::check(&d, &out)?;
    let a = out.model.audit(&d.request.input.observations)?;
    assert_eq!(a.train_errors, 0);
    assert!(a.train_ce < 1e-6 && a.gram_error < 1e-10 && a.coefficient_error < 1e-10);
    assert!(
        out.model
            .coefficients
            .iter()
            .enumerate()
            .any(|(i, x)| (i / out.model.vocabulary) % 2 == 1 && x.abs() > 0.1)
    );
    assert_eq!(out.bounds.states, out.observed_states + out.default_states);
    assert!(out.default_states <= d.names.rows.len() * d.names.columns.len());
    assert!(out.bounds.phase_entries < out.bounds.states.pow(2));
    assert!(out.unmatched_states > 0);
    Ok(())
}
#[test]
fn matched_unseen_code_keeps_the_readout_but_cannot_identify_a_missing_raw_alias() -> Result<()> {
    let d = alias_fixture()?;
    let out = initialize(&d)?;
    let readout = [0, 0, 1, 1, 0];
    assert!(out.model.known(8));
    assert_eq!(pooled(&out.model, 8, &readout), 0);
    let z = out.model.logits_cell(8);
    let best = (1..z.len()).fold(0, |b, c| if z[c] > z[b] { c } else { b });
    assert_eq!(best, 0);
    // Both raw0and the entirely unobserved raw4 have code/readout0.
    assert_ne!(best, 4);
    assert!(d.code.as_ref().unwrap().label(4).is_none());
    assert!(!d.raw_label_identification && !d.target_readout_compatibility_proved);
    Ok(())
}
#[test]
fn all_modes_replay_with_actual_tail_and_preserved_unseen_defaults() -> Result<()> {
    let d = fixture()?;
    let samples = &d.request.input.observations;
    let policies = training::policies(samples.len(), 2)?;
    let out = binding::run(&d, policies.clone())?;
    binding::replay(&d, &policies, &out)?;
    let initial = &out.initialization.as_ref().unwrap().model;
    for endpoint in &out.endpoints {
        let t = endpoint.training.as_ref().unwrap();
        assert_eq!(endpoint.audit.as_ref().unwrap().train_errors, 0);
        for cell in 0..9 {
            let (b, s) = initial.location(cell);
            if s + 1 == initial.blocks[b].states() {
                for (a, b) in t
                    .model
                    .logits_cell(cell)
                    .iter()
                    .zip(initial.logits_cell(cell))
                {
                    assert!((a - b).abs() < 1e-9);
                }
            }
        }
    }
    let mini = out.endpoints[1].training.as_ref().unwrap();
    assert_eq!(mini.work.gradient_examples, 5);
    assert_eq!(mini.steps.len(), 3);
    assert_eq!(mini.steps.last().unwrap().count, 1);
    let forward = out.endpoints[2].training.as_ref().unwrap();
    assert!(forward.steps.is_empty());
    assert_eq!(forward.work.gradient_batches, 0);
    assert_eq!(forward.work.head_derivative_terms, 0);
    assert_eq!(&forward.model, initial);
    Ok(())
}
#[test]
fn unmatched_states_and_unknown_axes_have_explicit_distinct_masks() -> Result<()> {
    let d = diagonal()?;
    let out = initialize(&d)?;
    assert_eq!(out.unmatched_states, 2);
    assert_eq!(out.unknown_axis_cells, 5);
    assert!(out.model.latent(1).is_some() && !out.model.known(1));
    assert!(out.model.latent(8).is_none() && !out.model.known(8));
    assert_eq!(
        out.model.logits_cell(8),
        out.model.logits_state(
            out.model.encoding.fallback_block,
            out.model.blocks[out.model.encoding.fallback_block].states() - 1
        )
    );
    assert!(out.model.encode(&[[1, 2]]).is_err());
    assert!(out.model.encode(&[[8, 2]]).is_err());
    assert!(out.model.encode(&[[0, 4]]).is_err());
    assert!(out.model.encode(&[[0, 2], [0, 2]]).is_err());
    Ok(())
}
#[test]
fn clipping_and_mutated_tensors_states_masks_and_claims_are_checked() -> Result<()> {
    let d = fixture()?;
    let initial = initialize(&d)?;
    let mut m = initial.model.clone();
    for (i, x) in m.table.iter_mut().enumerate() {
        *x = if i % 2 == 0 { 1000. } else { -1000. };
    }
    m.project();
    m.validate()?;
    assert!(m.table.contains(&32.) && m.table.contains(&-32.));
    assert!(
        m.audit(&d.request.input.observations)?
            .column_l1
            .iter()
            .all(|&x| x < m.radius)
    );
    for kind in 0..10 {
        let mut bad = initial.clone();
        match kind {
            0 => bad.model.coefficients[0] += 0.25,
            1 => bad.model.blocks[0].code[0] += C::new(0., 0.1),
            2 => bad.model.table[0] += 0.25,
            3 => bad.model.encoding.cell_blocks[0] = None,
            4 => bad.model.encoding.classes[0].clear(),
            5 => bad.model.encoding.keys[0].push(0),
            6 => bad.model.blocks[0].observed_labels[0] = 6,
            7 => bad.observed_states -= 1,
            8 => bad.ce_gradients = 1,
            _ => bad.model.encoding.code.dimension = 0,
        }
        assert!(construct::check(&d, &bad).is_err(), "state mutation{kind}");
    }
    let p = training::policies(5, 2)?;
    let out = binding::run(&d, p.clone())?;
    for kind in 0..5 {
        let mut bad = out.clone();
        match kind {
            0 => bad.raw_label_identification = true,
            1 => bad.target_readout_compatibility_proved = true,
            2 => bad.discovery_sha256.push('0'),
            3 => bad.endpoints.swap(0, 2),
            _ => bad.endpoints[2].audit.as_mut().unwrap().train_ce += 1.,
        }
        assert!(binding::check(&d, &p, &bad).is_err());
    }
    Ok(())
}
#[test]
fn preallocation_work_and_mode_refusals_preserve_available_modes() -> Result<()> {
    assert!(limits::shape(1025, &[1]).is_err());
    assert!(limits::shape(7, &[514]).is_err());
    assert!(limits::shape(1024, &[513; 257]).is_err());
    assert!(
        limits::shape(2, &[400; 8])
            .unwrap_err()
            .to_string()
            .contains("audit-work")
    );
    assert!(limits::dimensions(usize::MAX, 7).is_err());
    let d = fixture()?;
    let mut p = training::policies(5, 2)?;
    p[0].updates = 4097;
    let out = binding::run(&d, p.clone())?;
    assert!(out.endpoints[0].training.is_none() && out.endpoints[0].refusal.is_some());
    assert!(out.endpoints[1].training.is_some() && out.endpoints[2].training.is_some());
    binding::check(&d, &p, &out)?;
    Ok(())
}
#[test]
fn serialized_basis_actual_tensors_and_step_hashes_replay() -> Result<()> {
    let d = fixture()?;
    let restored: Report = serde_json::from_slice(&serde_json::to_vec(&d)?)?;
    assert_eq!(restored, d);
    discovery::replay(&d.request, &restored)?;
    let p = training::policies(5, 2)?;
    let out = binding::run(&d, p.clone())?;
    let restored: binding::Report = serde_json::from_slice(&serde_json::to_vec(&out)?)?;
    assert_eq!(restored, out);
    binding::replay(&d, &p, &restored)?;
    Ok(())
}

use crate::*;
use discovery::{Report, Request};
mod mutations;
mod resources;
fn fixture() -> Request {
    Request {
        input: kernel::discovery::Input {
            n: 2,
            vocabulary: 4,
            observations: vec![[0, 0], [1, 2], [2, 3]],
            policy: kernel::discovery::Policy::default(),
        },
        readout: vec![0, 0, 1, 1],
        answers: 2,
        prime: 2,
    }
}
fn report() -> Result<Report> {
    discovery::run(fixture())
}
fn best(z: &[f64]) -> usize {
    (1..z.len()).fold(0, |a, b| if z[b] > z[a] { b } else { a })
}
#[test]
fn cm35_unseen_corner_is_grouped_identified_while_raw_alias_is_wrong() -> Result<()> {
    let request = fixture();
    let old = kernel::discovery::run(kernel::discovery::Request {
        input: request.input.clone(),
        prime: 2,
    })?;
    kernel::discovery::replay(&old.request, &old)?;
    let old_code = old.code.as_ref().unwrap();
    assert_eq!(old_code.dimension, 2);
    assert!(
        old_code
            .names
            .classes
            .iter()
            .all(|&c| old_code.query(1, 1).as_deref() != old_code.label(c))
    );
    let out = report()?;
    discovery::replay(&request, &out)?;
    let grouped = out.grouped.as_ref().unwrap();
    assert_eq!(grouped.request.input.vocabulary, 2);
    assert_eq!(
        grouped.request.input.observations,
        vec![[0, 0], [1, 1], [2, 1]]
    );
    assert_eq!(grouped.field.as_ref().unwrap().problem.system.rows.len(), 5);
    assert!(grouped.basis_audit.as_ref().unwrap().complete);
    assert_eq!(grouped.code.as_ref().unwrap().dimension, 1);
    let raw = out.raw_code.as_ref().unwrap();
    assert_eq!(raw.names.classes, vec![0, 2, 3]);
    assert_eq!(raw.query(1, 1).as_deref(), raw.label(0));
    assert_eq!(raw.label(2), raw.label(3));
    assert_eq!(out.work.grouped_observations, 3);
    assert_eq!(out.work.observed_vector_checks, 3);
    let initial = construct::initialize(&out)?;
    construct::check(&out, &initial)?;
    assert_eq!(initial.model.vocabulary, 4);
    assert!(initial.model.known(3));
    let logits = initial.model.logits_cell(3);
    let predicted = best(&logits);
    assert_eq!(predicted, 0);
    assert_ne!(predicted, 1); // Raw completion1 is equally consistent with q.
    assert_eq!(request.readout[predicted], request.readout[1]);
    assert!(logits[0] > logits[1] + 23.999);
    assert!(!out.raw_label_identification && !out.unknown_target_compatibility_proved);
    Ok(())
}
#[test]
fn original_ce_and_actual_complex_updates_keep_all_modes_and_minibatch_tail() -> Result<()> {
    let d = report()?;
    let policies = training::policies(3, 2)?;
    let bound = binding::run(&d, policies.clone())?;
    binding::replay(&d, &policies, &bound)?;
    let initial = &bound.initialization.as_ref().unwrap().model;
    for endpoint in &bound.endpoints {
        let t = endpoint.training.as_ref().unwrap();
        assert_eq!(t.model.vocabulary, 4);
        assert_eq!(endpoint.audit.as_ref().unwrap().train_errors, 0);
        assert!((t.initial_ce - (1. + 3. * (-24_f64).exp()).ln()).abs() < 1e-13);
        assert!(t.initial_ce > (1. + (-24_f64).exp()).ln() * 2.9);
        // Fixed table defaults are exact; the actual Fourier product rounds.
        for block in &t.model.blocks {
            let at = (block.offset + block.states() - 1) * t.model.vocabulary;
            assert_eq!(&t.model.table[at..at + 4], &initial.table[at..at + 4]);
        }
        assert!(
            t.model
                .logits_cell(3)
                .iter()
                .zip(initial.logits_cell(3))
                .all(|(&a, b)| (a - b).abs() < 1e-12)
        );
        assert_eq!(
            best(&audit::complex_logits(&t.model, t.model.location(3))),
            0
        );
        for cell in 0..4 {
            let a = t.model.logits_cell(cell);
            let b = audit::complex_logits(&t.model, t.model.location(cell));
            assert!(a.iter().zip(b).all(|(&x, y)| (x - y).abs() < 1e-12));
        }
    }
    assert_eq!(
        bound.endpoints[0]
            .training
            .as_ref()
            .unwrap()
            .work
            .gradient_batches,
        1
    );
    let mini = bound.endpoints[1].training.as_ref().unwrap();
    assert_eq!(
        mini.steps
            .iter()
            .map(|s| (s.start, s.count))
            .collect::<Vec<_>>(),
        vec![(0, 2), (2, 1)]
    );
    assert_eq!(mini.work.gradient_examples, 3);
    let forward = bound.endpoints[2].training.as_ref().unwrap();
    assert_eq!(forward.work.gradient_batches, 0);
    assert_eq!(forward.work.head_derivative_terms, 0);
    assert_eq!(forward.model, *initial);
    assert!(!bound.raw_label_identification && !bound.target_readout_compatibility_proved);
    Ok(())
}
#[test]
fn arbitrary_label_and_answer_permutations_transport_actual_predictions() -> Result<()> {
    let original = report()?;
    let a = construct::initialize(&original)?;
    for permutation in [[3, 0, 2, 1], [2, 3, 1, 0], [1, 2, 0, 3]] {
        let mut request = fixture();
        for (old, &new) in permutation.iter().enumerate() {
            request.readout[new] = 1 - original.request.readout[old];
        }
        for row in &mut request.input.observations {
            row[1] = permutation[row[1]];
        }
        let d = discovery::run(request.clone())?;
        discovery::replay(&request, &d)?;
        let b = construct::initialize(&d)?;
        construct::check(&d, &b)?;
        for cell in 0..4 {
            assert_eq!(
                best(&b.model.logits_cell(cell)),
                permutation[best(&a.model.logits_cell(cell))]
            );
            assert_eq!(a.model.known(cell), b.model.known(cell));
        }
        let policies = training::policies(3, 2)?;
        let bound = binding::run(&d, policies.clone())?;
        binding::replay(&d, &policies, &bound)?;
    }
    Ok(())
}
#[test]
fn unknown_axes_and_unmatched_active_codes_are_separate() -> Result<()> {
    let mut request = fixture();
    request.input.n = 3;
    request.input.observations = vec![[0, 0], [4, 2]];
    let d = discovery::run(request)?;
    let initial = construct::initialize(&d)?;
    construct::check(&d, &initial)?;
    assert_eq!(d.raw_code.as_ref().unwrap().dimension, 2);
    assert_eq!(initial.unknown_axis_cells, 5);
    assert_eq!(initial.unmatched_states, 2);
    assert!(initial.model.latent(1).is_some());
    assert!(!initial.model.known(1));
    assert!(initial.model.latent(2).is_none());
    assert!(!initial.model.known(2));
    Ok(())
}
#[test]
fn unequal_readout_fiber_sizes_and_unobserved_raw_labels_are_valid_metadata() -> Result<()> {
    let mut request = fixture();
    request.input.vocabulary = 5;
    request.readout.push(0);
    let d = discovery::run(request.clone())?;
    discovery::replay(&request, &d)?;
    let initial = construct::initialize(&d)?;
    construct::check(&d, &initial)?;
    assert_eq!(initial.model.vocabulary, 5);
    assert_eq!(initial.model.encoding.code.names.classes, vec![0, 2, 3]);
    assert_eq!(best(&initial.model.logits_cell(3)), 0);
    Ok(())
}

#[test]
fn categorical_axis_renaming_transports_the_unseen_query_and_leaves_unknown_axes() -> Result<()> {
    let old = construct::initialize(&report()?)?;
    let mut request = fixture();
    request.input.n = 3;
    let row = [2, 0];
    let column = [1, 2];
    for sample in &mut request.input.observations {
        sample[0] = row[sample[0] % 2] + 3 * column[sample[0] / 2];
    }
    let d = discovery::run(request.clone())?;
    discovery::replay(&request, &d)?;
    let new = construct::initialize(&d)?;
    construct::check(&d, &new)?;
    for cell in 0..4 {
        let renamed = row[cell % 2] + 3 * column[cell / 2];
        assert_eq!(
            best(&new.model.logits_cell(renamed)),
            best(&old.model.logits_cell(cell))
        );
        assert_eq!(new.model.known(renamed), old.model.known(cell));
    }
    assert_eq!(new.unknown_axis_cells, 5);
    Ok(())
}

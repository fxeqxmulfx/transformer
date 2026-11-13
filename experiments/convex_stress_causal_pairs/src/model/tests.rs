use super::*;
fn argmax(z: &[f64]) -> usize {
    (1..z.len()).fold(0, |best, c| if z[c] > z[best] { c } else { best })
}
#[test]
fn actual_complex_head_inverts_full_vocabulary_states_and_marks_unknowns() -> Result<()> {
    let request = crate::tests::pair_request();
    let discovery = discovery::run(request.clone())?;
    let initial = initialize(&discovery)?;
    check_initialization(&discovery, &initial)?;
    let model = &initial.model;
    let audit = model.audit(&request.input)?;
    assert_eq!(audit.train_errors, 0);
    assert_eq!(audit.known_training_prefixes, 4);
    assert!(!audit.exact_real_certificate && !audit.unseen_identification);
    assert!((audit.code_nuclear[0] - 5.).abs() < 1e-12);
    assert!((audit.code_singular_min[0] - 1.).abs() < 1e-12);
    assert!((audit.code_singular_max[0] - 1.).abs() < 1e-12);
    let expected_ce = (1. + 6. * (-24f64).exp()).ln();
    assert!((audit.train_ce - expected_ce).abs() < 1e-14);
    assert!(audit.head_column_l1.iter().all(|&v| v <= model.radius));
    for s in &request.input.observations {
        assert_eq!(argmax(&model.logits_tokens(&s.tokens)), s.label);
        assert!(model.known(&s.tokens));
    }
    for tokens in [&[0, 2, 6][..], &[0, 0, 5][..]] {
        assert!(!model.known(tokens));
        assert_eq!(argmax(&model.logits_tokens(tokens)), 0);
    }
    assert_eq!(initial.ce_gradients, 0);
    // The imaginary head is used by prediction, not merely stored for audits.
    let mut changed = model.clone();
    let coordinate = (model.blocks[0].first_row + 3) * model.vocabulary + 3;
    changed.coefficients[coordinate] += 0.5;
    changed.validate()?;
    assert!((changed.logits_state(0, 1)[3] - model.logits_state(0, 1)[3]).abs() > 0.1);
    Ok(())
}
#[test]
fn presence_head_preserves_permutations_and_handles_longer_prefixes() -> Result<()> {
    let mut request = crate::tests::pair_request();
    request.input.observations = vec![
        Observation {
            tokens: vec![1, 1, 6],
            label: 5,
        },
        Observation {
            tokens: vec![0, 1, 6],
            label: 4,
        },
    ];
    let d = discovery::run(request)?;
    let model = initialize(&d)?.model;
    let expected = model.logits_tokens(&[0, 1, 6]);
    assert_eq!(model.logits_tokens(&[1, 0, 6]), expected);
    let mut long = vec![1; 64];
    long[53] = 0;
    long.push(6);
    assert_eq!(model.logits_tokens(&long), expected);
    assert!(model.known(&long));
    assert_eq!(argmax(&expected), 4);
    Ok(())
}
#[test]
fn coverage_and_flatness_do_not_identify_the_candidate_on_a_new_prefix() -> Result<()> {
    let mut request = crate::tests::pair_request();
    request.input.observations = vec![
        Observation {
            tokens: vec![0, 0, 6],
            label: 2,
        },
        Observation {
            tokens: vec![1, 1, 6],
            label: 3,
        },
    ];
    let discovery = discovery::run(request.clone())?;
    let presence = initialize(&discovery)?.model;
    let mut positional = presence.clone();
    let decoder = &mut positional.blocks[0].decoder;
    decoder.candidate = features::Candidate::Single(2 * request.input.vocabulary + 2);
    decoder.labels.swap(0, 1);
    for row in positional
        .coefficients
        .chunks_exact_mut(request.input.vocabulary)
    {
        row.swap(2, 3);
    }
    // These are two actual members of the same fixed-feature model family,
    // not a claim that the fixed presence-first learner selects both.
    for model in [&presence, &positional] {
        model.validate()?;
        assert_eq!(model.audit(&request.input)?.train_errors, 0);
        assert!(model.known(&[0, 1, 6]));
    }
    assert_ne!(
        argmax(&presence.logits_tokens(&[0, 1, 6])),
        argmax(&positional.logits_tokens(&[0, 1, 6]))
    );
    assert_eq!(presence.blocks[0].code, positional.blocks[0].code);
    Ok(())
}
#[test]
fn malformed_actual_complex_states_and_original_data_are_rejected() -> Result<()> {
    let request = crate::tests::pair_request();
    let discovery = discovery::run(request.clone())?;
    let initial = initialize(&discovery)?;
    for i in 0..7 {
        let mut bad = initial.clone();
        match i {
            0 => bad.model.blocks[0].code[1].im += 0.1,
            1 => bad.model.blocks[0].first_row += 1,
            2 => bad.model.coefficients[0] = f64::NAN,
            3 => bad.model.radius += 1.,
            4 => bad.model.blocks[0].decoder.labels[0] = 1,
            5 => bad.model.coefficients[0] = bad.model.radius * 2.,
            _ => bad.complex_head_additions += 1,
        }
        assert!(check_initialization(&discovery, &bad).is_err());
    }
    let mut bad = request;
    bad.input.observations[0].label = 0;
    assert!(discovery::replay(&bad, &discovery).is_err());
    assert!(memory_bound(1024, 65536, 1).is_err());
    assert!(memory_bound(2, 1, 4_000_001).is_err());
    Ok(())
}

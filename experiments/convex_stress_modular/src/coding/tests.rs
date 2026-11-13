use super::*;

fn fixture(q: usize) -> (Input, Vec<usize>) {
    let n = q + 2;
    let rows: Vec<_> = (0..n).map(|i| (q - 1) * (i + 2) % q).collect();
    let columns: Vec<_> = (0..n).map(|j| (q - 1) * (j + 1) % q).collect();
    // Arbitrary class identities and unused vocabulary entries. The learner
    // sees only observed cell/class pairs, never these generator coordinates.
    let decoder: Vec<_> = (0..q).map(|z| 3 * ((q - 1) * z % q) + 1).collect();
    let labels: Vec<_> = (0..n * n)
        .map(|cell| decoder[(rows[cell % n] + columns[cell / n]) % q])
        .collect();
    let observations = (0..n * n)
        .filter(|&cell| cell != n + 1 && cell != n * n - 2)
        .map(|cell| [cell, labels[cell]])
        .collect();
    (
        Input {
            n,
            vocab: 3 * q + 3,
            observations,
            policy: Policy::default(),
        },
        labels,
    )
}
fn checked(input: &Input) -> Report {
    let out = run(input.clone()).unwrap();
    check(input, &out).unwrap();
    replay(input, &out).unwrap();
    assert_eq!(
        serde_json::from_slice::<Report>(&serde_json::to_vec(&out).unwrap()).unwrap(),
        out
    );
    out
}

#[test]
fn masked_categorical_tables_recover_prime_powers_crt_and_repeated_phases() {
    for q in 2..=16 {
        let (input, labels) = fixture(q);
        let out = checked(&input);
        assert_eq!(
            out.reason,
            "admitted",
            "q={q}, components={:?}",
            out.components.iter().map(|c| &c.reason).collect::<Vec<_>>()
        );
        let model = out.model.as_ref().unwrap();
        assert_eq!(model.output_labels, labels, "q={q}");
        assert_eq!(model.rows[0], model.rows[q]);
        assert_eq!(model.columns[0], model.columns[q]);
        assert_eq!(out.audit.as_ref().unwrap()["training_errors"], 0);
        assert_eq!(out.work.ce_gradients, 0);
        assert_eq!(out.work.raw_frequencies, q);
        assert_eq!(
            out.work.finite_stages,
            out.factors.iter().map(|f| f.exponent).sum::<usize>()
        );
        assert_eq!(out.crt.len(), out.factors.len() - 1);
        assert!(
            out.components.iter().flat_map(|c| &c.lifts).all(|s| s
                .elimination
                .space
                .as_ref()
                .unwrap()
                .directions
                .is_empty())
        );
    }
}

#[test]
fn class_renaming_leaves_discovery_coordinates_and_unused_logits_intact() {
    for q in [6, 8, 12] {
        let (input, _) = fixture(q);
        let out = checked(&input);
        let mut renamed = input.clone();
        for s in &mut renamed.observations {
            s[1] = input.vocab - 1 - s[1];
        }
        let other = checked(&renamed);
        assert_eq!(out.point, other.point);
        assert_eq!(out.system, other.system);
        assert_eq!(out.components, other.components);
        assert_eq!(out.work, other.work);
        let original = out.model.as_ref().unwrap();
        let changed = other.model.as_ref().unwrap();
        assert_eq!(original.rows, changed.rows);
        assert_eq!(original.columns, changed.columns);
        assert!(
            original
                .output_labels
                .iter()
                .zip(&changed.output_labels)
                .all(|(&a, &b)| a + b == input.vocab - 1)
        );
        assert_eq!(other.audit.as_ref().unwrap()["vocabulary"], input.vocab);
    }
}

#[test]
fn missing_axes_are_recorded_choices_not_unseen_identification() {
    let (mut input, truth) = fixture(6);
    input.observations.retain(|s| s[0] % input.n != input.n - 1);
    let out = checked(&input);
    assert_eq!(out.partition.missing_rows, vec![input.n - 1]);
    assert_eq!(out.reason, "admitted");
    assert_eq!(out.point.as_ref().unwrap()[input.n - 1], 0);
    assert_ne!(out.model.as_ref().unwrap().output_labels, truth);
    assert_eq!(
        check(&input, &out).unwrap()["unseen_answers_identified"],
        false
    );
}

#[test]
fn every_stage_and_work_cap_is_a_retained_refusal() {
    let (original, _) = fixture(12);
    for which in 0..6 {
        let mut input = original.clone();
        match which {
            0 => input.policy.stages = 0,
            1 => input.policy.stages = 1,
            2 => input.policy.scalar_updates_per_stage = 0,
            3 => input.policy.total_scalar_updates = 0,
            4 => input.policy.row_operations_per_stage = 0,
            _ => input.policy.observation_equations = 1,
        }
        let out = checked(&input);
        assert_eq!(out.reason, "component_refused");
        assert!(out.model.is_none());
        assert_eq!(out.components.len(), 2);
        assert_eq!(out.components[1].reason, "earlier_component_refused");
        assert!(out.work.finite_stages <= input.policy.stages);
        assert!(out.work.finite_scalar_updates <= input.policy.total_scalar_updates);
    }
    let complete = checked(&original);
    let mut input = original;
    input.policy.total_scalar_updates = complete.components[0]
        .base
        .as_ref()
        .unwrap()
        .field
        .work
        .scalar_updates;
    let out = checked(&input);
    assert_eq!(out.components[0].lifts[0].reason, "scalar_update_cap");
}

#[test]
fn unsupported_domains_noncyclic_kernels_and_unretained_equations_are_explicit() {
    let (mut input, _) = fixture(3);
    for s in &mut input.observations {
        s[1] = 1;
    }
    assert_eq!(checked(&input).reason, "unsupported_latent_domain");
    let input = Input {
        n: 17,
        vocab: 300,
        observations: (0..258).map(|i| [i, i]).collect(),
        policy: Policy::default(),
    };
    assert_eq!(checked(&input).reason, "unsupported_latent_domain");
    let input = Input {
        n: 4,
        vocab: 7,
        observations: (0..16).map(|i| [i, (i % 4) ^ (i / 4)]).collect(),
        policy: Policy::default(),
    };
    let out = checked(&input);
    assert_eq!(out.reason, "component_refused");
    assert_eq!(out.components[0].reason, "nullity_cap");
    let mut input = Input {
        n: 4,
        vocab: 5,
        observations: (0..16).map(|i| [i, (i % 4 + i / 4) % 3]).collect(),
        policy: Policy::default(),
    };
    input.policy.observation_equations = 15;
    input.observations[15][1] = 1;
    let out = checked(&input);
    assert_eq!(out.reason, "original_observation_conflict");
    assert_eq!(out.observation_conflict, Some(15));
    assert_eq!(out.work.observation_checks, 16);
    assert!(out.model.is_none());
}

#[test]
fn distinct_classes_cannot_share_a_code_and_invalid_inputs_cannot_start() {
    let input = Input {
        n: 2,
        vocab: 3,
        observations: vec![[0, 0], [1, 1], [2, 0], [3, 2]],
        policy: Policy::default(),
    };
    let out = checked(&input);
    assert_eq!(out.reason, "class_code_collision");
    assert_eq!(out.collision, Some([1, 2, 1]));
    assert!(out.model.is_none());
    let mut bad = out.clone();
    bad.collision = None;
    assert!(check(&input, &bad).is_err());
    for which in 0..6 {
        let mut bad = input.clone();
        match which {
            0 => bad.observations[1][0] = 0,
            1 => bad.observations[0][1] = 3,
            2 => bad.observations.clear(),
            3 => bad.n = 258,
            4 => bad.policy.stages = 17,
            _ => bad.policy.total_scalar_updates = 4_000_000_001,
        }
        assert!(run(bad).is_err());
    }
}

#[test]
fn changed_integer_component_states_crt_endpoints_and_accounting_are_rejected() {
    let (input, _) = fixture(12);
    let out = checked(&input);
    for mutation in 0..14 {
        let mut bad = out.clone();
        match mutation {
            0 => bad.partition.classes.swap(0, 1),
            1 => bad.system.as_mut().unwrap().rows[0][0] += 1,
            2 => bad.components[0].base.as_mut().unwrap().inverse = Some(0),
            3 => {
                bad.components[0]
                    .base
                    .as_mut()
                    .unwrap()
                    .point
                    .as_mut()
                    .unwrap()[0] += 1
            }
            4 => bad.components[0].lifts[0].residual[0] += 1,
            5 => bad.components[0].lifts[0].next.as_mut().unwrap()[0] += 1,
            6 => bad.crt[0].bezout.as_mut().unwrap()[0] += 1,
            7 => bad.crt_audits[0]["unseen_answers_identified"] = true.into(),
            8 => bad.point.as_mut().unwrap()[0] += 1,
            9 => bad.model.as_mut().unwrap().output_labels[0] += 1,
            10 => bad.audit.as_mut().unwrap()["train_ce"] = 0.into(),
            11 => bad.work.finite_scalar_updates += 1,
            12 => bad.work.ce_gradients = 1,
            _ => bad.components[0].factor.exponent += 1,
        }
        assert!(check(&input, &bad).is_err(), "mutation={mutation}");
    }
    let mut changed = input.clone();
    changed.observations[0][1] = (changed.observations[0][1] + 1) % input.vocab;
    assert!(check(&changed, &out).is_err());
    assert!(replay(&changed, &out).is_err());
    let mut bad = out;
    bad.work.observation_checks += 1;
    assert!(replay(&input, &bad).is_err());
}

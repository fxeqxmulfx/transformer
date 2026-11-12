use crate::*;

fn obs(row: usize, column: usize, label: usize) -> Observation {
    Observation { row, column, label }
}

fn wide() -> (Problem, Config) {
    let p = Problem {
        side: 2,
        states: 4,
        vocab: 2,
        decoders: vec![vec![0, 0, 0, 1]],
        observations: vec![obs(0, 0, 1), obs(0, 1, 1), obs(1, 0, 0)],
    };
    let c = Config::new(p.observations[2], 2);
    (p, c)
}

fn complete(p: &Problem, c: Config) -> Certificate {
    match propagate(p, c).unwrap() {
        Outcome::Complete(saved) => *saved,
        other => panic!("unexpected refusal: {other:?}"),
    }
}

#[test]
fn lean_wide_fiber_identifies_output_but_not_latent_state() {
    let (p, c) = wide();
    let saved = complete(&p, c);
    let checked = verify(&p, c, &saved).unwrap();
    assert_eq!(checked.answer(1, 1), Some(0));
    assert!(!checked.inconsistent());
    assert_eq!(saved.branches.len(), 3);
    for (z, branch) in saved.branches.iter().enumerate() {
        assert!(branch.live);
        assert_eq!(branch.domains.rows[1], vec![0]);
        assert_eq!(branch.domains.columns[1], vec![z]);
        assert_eq!(branch.domains.rows[0], vec![(3 + 4 - z) % 4]);
    }
    assert!(!p.observations.iter().any(|t| t.row == 1 && t.column == 1));
}

#[test]
fn both_shifted_bank_members_contribute() {
    let (mut p, c) = wide();
    p.decoders.push(vec![0, 0, 1, 0]);
    let saved = complete(&p, c);
    assert_eq!(saved.branches.len(), 6);
    assert_eq!(verify(&p, c, &saved).unwrap().answer(1, 1), Some(0));
    p.decoders.reverse();
    assert_eq!(complete(&p, c).answers, saved.answers);
}

#[test]
fn individually_identified_decoders_can_disagree_and_must_be_unioned() {
    let observations = vec![obs(0, 0, 0), obs(0, 1, 1), obs(1, 0, 2)];
    let mut p = Problem {
        side: 2,
        states: 4,
        vocab: 4,
        decoders: vec![vec![0, 1, 2, 3]],
        observations,
    };
    let c = Config::new(p.observations[0], 2);
    assert_eq!(
        verify(&p, c, &complete(&p, c)).unwrap().answer(1, 1),
        Some(3)
    );
    p.decoders = vec![vec![0, 1, 3, 2]];
    assert_eq!(
        verify(&p, c, &complete(&p, c)).unwrap().answer(1, 1),
        Some(0)
    );
    p.decoders.push(vec![0, 1, 2, 3]);
    let saved = complete(&p, c);
    assert_eq!(saved.answers[3], vec![0, 3]);
    assert_eq!(verify(&p, c, &saved).unwrap().answer(1, 1), None);
}

#[test]
fn one_latent_state_and_unobserved_coordinates() {
    let p = Problem {
        side: 2,
        states: 1,
        vocab: 3,
        decoders: vec![vec![2]],
        observations: vec![obs(1, 0, 2)],
    };
    for steps in [0, 1, 4] {
        let c = Config::new(p.observations[0], steps);
        let saved = complete(&p, c);
        assert_eq!(saved.answers, vec![vec![2]; 4]);
        assert_eq!(verify(&p, c, &saved).unwrap().answer(0, 1), Some(2));
    }
}

#[test]
fn ambiguity_and_cross_decoder_disagreement_are_refused() {
    let (mut p, mut c) = wide();
    for t in &mut p.observations {
        t.label = 0;
    }
    c.anchor = p.observations[0];
    for bank in [
        vec![vec![0, 0, 0, 1]],
        vec![vec![0, 0, 0, 1], vec![0, 1, 0, 1]],
    ] {
        p.decoders = bank;
        for steps in [0, 1, 2, 5, 20] {
            c.steps = steps;
            let saved = complete(&p, c);
            assert_eq!(saved.answers[3], vec![0, 1]);
            assert_eq!(verify(&p, c, &saved).unwrap().answer(1, 1), None);
        }
    }
}

#[test]
fn contradictions_and_labels_outside_the_image_have_empty_unions() {
    let (mut p, mut c) = wide();
    p.side = 1;
    p.decoders.push(vec![0, 0, 1, 0]);
    p.observations = vec![obs(0, 0, 1), obs(0, 0, 0)];
    c.anchor = p.observations[0];
    c.steps = 1;
    let saved = complete(&p, c);
    assert!(saved.branches.iter().all(|b| !b.live));
    assert!(verify(&p, c, &saved).unwrap().inconsistent());
    p.decoders = vec![vec![0, 0, 0, 0]];
    c.steps = 0;
    let saved = complete(&p, c);
    assert!(saved.branches.is_empty());
    assert!(verify(&p, c, &saved).unwrap().inconsistent());
}

#[test]
fn input_permutations_output_renaming_and_unused_classes() {
    let (mut p, mut c) = wide();
    p.vocab = 5;
    for d in &mut p.decoders {
        for y in d {
            *y = [4, 2][*y];
        }
    }
    for t in &mut p.observations {
        t.row = 1 - t.row;
        t.column = 1 - t.column;
        t.label = [4, 2][t.label];
    }
    c.anchor = p.observations[2];
    let saved = complete(&p, c);
    let checked = verify(&p, c, &saved).unwrap();
    assert_eq!(checked.answer(0, 0), Some(4));
    assert!(saved.answers.iter().flatten().all(|&y| y == 2 || y == 4));
    assert_eq!(checked.answer(usize::MAX, 0), None);
}

#[test]
fn observations_retain_repetitions_and_are_order_and_batch_invariant() {
    let (p, c) = wide();
    let baseline = complete(&p, c);
    let mut reversed = p.clone();
    reversed.observations.reverse();
    assert_eq!(complete(&reversed, c), baseline);
    for size in 1..=p.observations.len() {
        let mut accumulated = p.clone();
        accumulated.observations.clear();
        for chunk in p.observations.chunks(size) {
            accumulated.observations.extend_from_slice(chunk);
        }
        assert_eq!(complete(&accumulated, c), baseline);
    }
    let mut repeated = p.clone();
    repeated.observations.extend_from_slice(&p.observations);
    let saved = complete(&repeated, c);
    assert_eq!(saved.answers, baseline.answers);
    assert_eq!(saved.problem.observations.len(), 6);
    assert!(saved.costs.support_checks > baseline.costs.support_checks);
    assert!(verify(&p, c, &saved).is_err());
    verify(&repeated, c, &saved).unwrap();
}

#[test]
fn budgets_never_expose_a_partial_union_as_a_certificate() {
    let (p, mut c) = wide();
    let full = complete(&p, c);
    c.max_checks = full.costs.support_checks + full.costs.output_checks - 1;
    let Outcome::Refused(r) = propagate(&p, c).unwrap() else {
        panic!("partial union certified");
    };
    assert_eq!(r.reason, RefusalReason::WorkBudget);
    assert!(r.completed_branches > 0 && r.completed_branches < r.total_branches);
    c.max_checks = 0;
    assert!(matches!(
        propagate(&p, c).unwrap(),
        Outcome::Refused(Refusal {
            reason: RefusalReason::WorkBudget,
            ..
        })
    ));
    c.max_branches = 2;
    let Outcome::Refused(r) = propagate(&p, c).unwrap() else {
        panic!("branch omitted");
    };
    assert_eq!(r.completed_branches, 0);
    assert_eq!(
        r.reason,
        RefusalReason::BranchBudget {
            required: 3,
            limit: 2
        }
    );
    c = Config::new(c.anchor, 2);
    c.memory_limit = preflight(&p, c).unwrap().memory_bound - 1;
    assert!(matches!(
        propagate(&p, c).unwrap(),
        Outcome::Refused(Refusal {
            reason: RefusalReason::Memory { .. },
            ..
        })
    ));
}

#[test]
fn malformed_and_oversized_requests_are_rejected_before_dense_allocation() {
    let (p, c) = wide();
    let mut invalid = p.clone();
    invalid.side = usize::MAX;
    assert!(propagate(&invalid, c).is_err());
    invalid = p.clone();
    invalid.states = 0;
    assert!(propagate(&invalid, c).is_err());
    invalid = p.clone();
    invalid.decoders[0][1] = p.vocab;
    assert!(propagate(&invalid, c).is_err());
    invalid = p.clone();
    invalid.observations[0].column = p.side;
    assert!(propagate(&invalid, c).is_err());
    invalid = p.clone();
    invalid.observations.clear();
    assert!(propagate(&invalid, c).is_err());
    assert!(
        propagate(
            &p,
            Config {
                steps: usize::MAX,
                ..c
            }
        )
        .is_err()
    );
    assert!(
        propagate(
            &p,
            Config {
                memory_limit: u64::MAX,
                ..c
            }
        )
        .is_err()
    );
    assert!(
        propagate(
            &p,
            Config {
                anchor: obs(1, 1, 0),
                ..c
            }
        )
        .is_err()
    );
    invalid = p.clone();
    invalid.side = 4096;
    invalid.vocab = 65536;
    assert!(preflight(&invalid, c).unwrap().memory_bound > RUN_MEMORY_BYTES);
    assert!(matches!(
        propagate(&invalid, c).unwrap(),
        Outcome::Refused(_)
    ));
}

#[test]
fn saved_evidence_roundtrips_and_tampering_is_rejected() {
    let (p, c) = wide();
    let saved = complete(&p, c);
    let bytes = serde_json::to_vec(&saved).unwrap();
    let decoded: Certificate = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(saved, decoded);
    verify(&p, c, &decoded).unwrap();
    for tamper in 0..11 {
        let mut bad = saved.clone();
        match tamper {
            0 => {
                bad.branches.pop();
            }
            1 => bad.branches[0].anchor_state = 3,
            2 => bad.branches[0].domains.rows[0].clear(),
            3 => bad.branches[0].live = false,
            4 => bad.branches[0].removed_per_sweep[0] += 1,
            5 => bad.answers[3] = vec![1],
            6 => bad.costs.support_checks += 1,
            7 => bad.problem.observations[0].label ^= 1,
            8 => bad.config.steps += 1,
            9 => bad.memory_bound -= 1,
            _ => bad.branches[0].decoder = 1,
        }
        assert!(verify(&p, c, &bad).is_err(), "accepted tamper {tamper}");
    }
    let mut value = serde_json::to_value(&saved).unwrap();
    value["trusted_coverage"] = true.into();
    assert!(serde_json::from_value::<Certificate>(value).is_err());
}

mod exhaustive;

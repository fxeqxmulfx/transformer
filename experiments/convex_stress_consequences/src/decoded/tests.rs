use super::*;

fn fixture(q: usize, masked: bool) -> Request {
    let n = q;
    let predictions: Vec<_> = (0..n * n)
        .map(|cell| (2 * (cell % n) + cell / n + 1) % q + 3)
        .collect();
    let observations = predictions
        .iter()
        .enumerate()
        .filter(|&(cell, _)| {
            let (i, j) = (cell % n, cell / n);
            (i, j) != (n - 1, n - 1)
                && (!masked || i == 0 || j == 0 || (cell * 31 + cell * cell * 7) % 101 > 49)
        })
        .map(|(cell, &label)| [cell, label])
        .collect();
    Request {
        n,
        vocabulary: q + 7,
        modulus: q as u16,
        observations,
        predictions,
        prefix_policy: row_program::Policy::default(),
        policy: Policy::default(),
    }
}
fn checked(r: &Request) -> Report {
    let out = run(r.clone()).unwrap();
    check(r, &out).unwrap();
    replay(r, &out).unwrap();
    assert_eq!(
        serde_json::from_slice::<Report>(&serde_json::to_vec(&out).unwrap()).unwrap(),
        out
    );
    out
}
#[test]
fn noncyclic_composite_witness_and_masked_tables_have_actual_query_programs() {
    for q in [4usize, 6, 8, 16, 17, 32] {
        let r = fixture(q, q != 4);
        let out = checked(&r);
        assert!(
            out.certified,
            "q={q}; refused={:?}",
            out.queries
                .iter()
                .filter(|x| x.decision != Decision::Passed)
                .map(|x| (x.cell, &x.decision))
                .collect::<Vec<_>>()
        );
        assert_eq!(out.prefix.input.rows.len(), r.observations.len());
        assert_eq!(out.queries.len(), q * q);
        assert_eq!(out.work.ce_gradients, 0);
        assert!(out.queries.last().unwrap().operations.len() > 1);
        assert!(out.prefix.pivots.len() < out.prefix.input.variables);
    }
}
#[test]
fn missing_axis_and_wrong_unseen_answers_are_never_certified() {
    let mut missing = fixture(4, false);
    missing.observations.retain(|s| s[0] % 4 != 3);
    let out = checked(&missing);
    assert!(!out.certified);
    for query in &out.queries {
        if query.cell % 4 == 3 {
            assert_ne!(query.decision, Decision::Passed);
        } else {
            assert_eq!(query.decision, Decision::Passed);
        }
    }
    let mut wrong = fixture(4, false);
    wrong.predictions[15] = 3 + (wrong.predictions[15] - 3 + 1) % 4;
    let out = checked(&wrong);
    assert!(!out.certified);
    assert_ne!(out.queries[15].decision, Decision::Passed);
    wrong.predictions[15] = 0;
    let out = checked(&wrong);
    assert_eq!(out.queries[15].decision, Decision::UnknownClass);
}

#[test]
fn all432_binary_realizable_observation_and_query_checks_are_sound() {
    let mut count = 0;
    for phases in 0..8 {
        let target: Vec<_> = (0..4)
            .map(|cell| (((cell % 2) * (phases & 1)) + ((phases >> (1 + cell / 2)) & 1)) % 2)
            .collect();
        for mask in 1..16 {
            let observations: Vec<_> = target
                .iter()
                .enumerate()
                .filter(|&(cell, _)| mask & (1 << cell) != 0)
                .map(|(cell, &label)| [cell, label])
                .collect();
            if !(0..2).all(|label| observations.iter().any(|s| s[1] == label)) {
                continue;
            }
            for cell in 0..4 {
                for label in 0..2 {
                    let mut r = fixture(2, false);
                    r.vocabulary = 2;
                    r.observations = observations.clone();
                    r.predictions = target.clone();
                    r.predictions[cell] = label;
                    let out = checked(&r);
                    for query in &out.queries {
                        if query.decision == Decision::Passed {
                            assert_eq!(query.label, target[query.cell]);
                        }
                    }
                    if observations.len() >= 3 && label == target[cell] {
                        assert!(out.certified);
                    }
                    count += 1;
                }
            }
        }
    }
    assert_eq!(count, 432);
}
#[test]
fn every_query_budget_and_observation_prefix_remains_explicit() {
    for slot in 0..4 {
        let mut r = fixture(4, false);
        match slot {
            0 => r.policy.queries = 0,
            1 => r.policy.reduction_products = 0,
            2 => r.policy.certificate_operations = 0,
            _ => r.policy.certificate_products = 0,
        }
        let out = checked(&r);
        assert!(!out.certified);
        assert!(matches!(
            out.reason.as_str(),
            "query_count_cap" | "query_work_cap"
        ));
    }
    let mut r = fixture(4, false);
    r.policy.equations = 3;
    let out = checked(&r);
    assert_eq!(out.prefix.input.rows.len(), 3);
    assert!(!out.certified);
    let mut r = fixture(4, false);
    r.prefix_policy.scalar_updates = 0;
    let out = checked(&r);
    assert!(!out.certified);
    assert_eq!(out.prefix.reason, "scalar_update_cap");
}
#[test]
fn altered_observed_matrices_query_programs_labels_and_coverage_are_rejected() {
    let r = fixture(4, false);
    let out = checked(&r);
    let last = out.queries.len() - 1;
    for slot in 0..14 {
        let mut bad = out.clone();
        match slot {
            0 => bad
                .prefix
                .input
                .rows
                .push(vec![0; bad.prefix.input.variables]),
            1 => bad.prefix.matrix[0][0] = 0,
            2 => bad.queries[last].label = 0,
            3 => bad.queries[last].operations.clear(),
            4 => bad.queries[last].receiver = Some(usize::MAX),
            5 => bad.queries[last].terms[0].factor = 0,
            6 => bad.queries[last].residual[0] = 1,
            7 => bad.queries[last].decision = Decision::Unrepresented,
            8 => bad.classes.swap(0, 1),
            9 => bad.certified = false,
            10 => bad.work.reduction_products -= 1,
            11 => bad.work.certificate_products -= 1,
            12 => bad.work.ce_gradients = 1,
            _ => {
                bad.queries.pop();
            }
        }
        assert!(check(&r, &bad).is_err(), "mutation {slot}");
        assert!(replay(&r, &bad).is_err());
    }
}

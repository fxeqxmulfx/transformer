use super::*;
use convex_stress_modular::Model;
fn fixture(q: usize) -> Request {
    let n = q;
    let variables = 3 * n;
    let mut rows = vec![];
    // CM29's repeated-row witness. Numeric names appear only in this fixture,
    // never in component discovery, which receives actual coefficient rows.
    for j in 0..n {
        for i in 0..n {
            if (i, j) == (n - 1, n - 1) {
                continue;
            }
            let mut row = vec![0; variables];
            row[i] = 1;
            row[n + j] = 1;
            row[2 * n + (2 * i + j) % q] = -1;
            rows.push(row);
        }
    }
    for coordinate in [0, 2 * n] {
        let mut row = vec![0; variables];
        row[coordinate] = 1;
        rows.push(row);
    }
    let right = vec![0; rows.len()];
    Request {
        n,
        classes: q,
        factor: Factor {
            prime: 2,
            exponent: q.ilog2() as usize,
            modulus: q as u16,
        },
        system: System {
            variables,
            rows,
            right,
        },
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
fn predictions(r: &Request, point: &[u16]) -> Vec<usize> {
    let q = r.classes;
    let codes = &point[2 * r.n..];
    let decoder = (0..q)
        .map(|z| codes.iter().position(|&c| c as usize == z).unwrap())
        .collect();
    Model::construct(
        q,
        point[..r.n].iter().map(|&x| x as usize).collect(),
        point[r.n..2 * r.n].iter().map(|&x| x as usize).collect(),
        decoder,
    )
    .unwrap()
    .output_labels
}
#[test]
fn noncyclic_composite_witnesses_keep_distinct_codes_with_identical_decoding() {
    for q in [4usize, 8, 16, 32, 64] {
        let r = fixture(q);
        let out = checked(&r);
        assert_eq!(
            out.base
                .as_ref()
                .unwrap()
                .elimination
                .space
                .as_ref()
                .unwrap()
                .directions
                .len(),
            2
        );
        assert!(!out.complete.is_empty(), "no final component for q={q}");
        if q > 4 {
            assert!(out.complete.len() >= 2, "no distinct codes for q={q}");
        }
        let target: Vec<_> = (0..q * q)
            .map(|cell| (2 * (cell % q) + cell / q) % q)
            .collect();
        for &i in &out.complete {
            assert_eq!(predictions(&r, &out.states[i].point), target);
        }
        if q <= 16 {
            assert!(out.stages.iter().any(|s| s.report.branches.is_empty()));
        }
        assert!(
            out.states
                .iter()
                .any(|s| matches!(s.decision, Decision::Multiplicity { .. }))
        );
        assert_eq!(out.work.ce_gradients, 0);
    }
}
#[test]
fn missing_axis_keeps_incompatible_decodings_without_identifying_one() {
    let mut r = fixture(4);
    r.system.rows.retain(|row| row[3] == 0);
    r.system.right = vec![0; r.system.rows.len()];
    let out = checked(&r);
    let tables: std::collections::BTreeSet<_> = out
        .complete
        .iter()
        .map(|&i| predictions(&r, &out.states[i].point))
        .collect();
    assert!(tables.len() > 1);
    for table in tables {
        for j in 0..4 {
            for i in 0..3 {
                assert_eq!(table[i + 4 * j], (2 * i + j) % 4);
            }
        }
    }
}
#[test]
fn caps_preserve_branch_sources_and_never_admit_a_partial_modulus() {
    let r = fixture(8);
    let full = checked(&r);
    for cap in [0, 1, 2, 4] {
        let mut r = r.clone();
        r.policy.stages = cap;
        let out = checked(&r);
        assert!(out.work.stages <= cap);
        for &s in &out.complete {
            assert_eq!(out.states[s].modulus, 8);
        }
        if cap < 4 {
            assert!(out.complete.is_empty());
        }
    }
    let mut r = r.clone();
    r.policy.states = 0;
    let out = checked(&r);
    assert!(out.complete.is_empty());
    assert!(out.states.iter().any(|s| s.decision == Decision::StateCap));
    let mut r = fixture(8);
    r.policy.enumeration.points = 1;
    let out = checked(&r);
    assert!(out.complete.is_empty());
    assert_eq!(out.base.as_ref().unwrap().reason, "point_cap");
    let mut r = fixture(8);
    r.policy.total_scalar_updates = full.base.as_ref().unwrap().elimination.work.scalar_updates;
    let out = checked(&r);
    assert!(out.complete.is_empty());
    assert!(out.states.iter().any(|s| s.decision == Decision::ScalarCap));
}
#[test]
fn prime_normalization_retains_zero_and_projective_duplicate_proposals() {
    let mut r = fixture(4);
    r.n = 3;
    r.classes = 3;
    r.factor = Factor {
        prime: 3,
        exponent: 1,
        modulus: 3,
    };
    let mut rows = vec![];
    for j in 0..3 {
        for i in 0..3 {
            let mut row = vec![0; 9];
            row[i] = 1;
            row[3 + j] = 1;
            row[6 + (i + j) % 3] = -1;
            rows.push(row);
        }
    }
    for c in [0, 6] {
        let mut row = vec![0; 9];
        row[c] = 1;
        rows.push(row);
    }
    r.system = System {
        variables: 9,
        right: vec![0; rows.len()],
        rows,
    };
    let out = checked(&r);
    assert_eq!(out.complete.len(), 1);
    assert!(
        out.states
            .iter()
            .any(|s| s.decision == Decision::NoClassUnit)
    );
    assert!(
        out.states
            .iter()
            .any(|s| matches!(s.decision, Decision::Duplicate { .. }))
    );
    assert!(out.stages.is_empty());
    assert_eq!(out.work.stages, 1);
}
#[test]
fn mutations_of_actual_points_parents_filters_and_work_are_rejected() {
    let r = fixture(8);
    let out = checked(&r);
    let complete = out.complete[0];
    for slot in 0..15 {
        let mut bad = out.clone();
        match slot {
            0 => bad.states[complete].point[0] = 1,
            1 => bad.states[complete].modulus = 4,
            2 => bad.states[complete].counts[0] += 1,
            3 => bad.states[complete].decision = Decision::StateCap,
            4 => bad.stages[0].parent = 0,
            5 => bad.stages[0]
                .report
                .input
                .system
                .right
                .last_mut()
                .map(|b| *b = 0)
                .unwrap(),
            6 => bad.complete.clear(),
            7 => bad.work.endpoint_products -= 1,
            8 => bad.work.class_coordinates -= 1,
            9 => bad.work.ce_gradients = 1,
            10 => bad.states[1].inverse = Some(0),
            11 => {
                bad.states.pop();
            }
            12 => bad.states.clear(),
            13 => bad.stages[0].parent = usize::MAX,
            _ => {
                bad.states[out.stages[0].parent].anchor = Some(usize::MAX);
            }
        }
        assert!(check(&r, &bad).is_err(), "mutation{slot}");
        assert!(replay(&r, &bad).is_err());
    }
    let mut invalid = r;
    invalid.factor.modulus = 16;
    assert!(run(invalid).is_err());
}

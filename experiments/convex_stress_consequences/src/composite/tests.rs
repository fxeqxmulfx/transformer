use super::*;
use convex_stress_modular::Model;

fn fixture(q: usize) -> Request {
    let mut rows = vec![];
    for j in 0..q {
        for i in 0..q {
            if (i, j) == (q - 1, q - 1) {
                continue;
            }
            let mut row = vec![0; 3 * q];
            row[i] = 1;
            row[q + j] = 1;
            row[2 * q + (2 * i + j) % q] = -1;
            rows.push(row);
        }
    }
    for coordinate in [0, 2 * q] {
        let mut row = vec![0; 3 * q];
        row[coordinate] = 1;
        rows.push(row);
    }
    Request {
        n: q,
        classes: q,
        system: System {
            variables: 3 * q,
            right: vec![0; rows.len()],
            rows,
        },
        component_policy: component::Policy::default(),
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
    let codes = &point[2 * r.n..];
    let decoder = (0..r.classes)
        .map(|z| codes.iter().position(|&c| c as usize == z).unwrap())
        .collect();
    Model::construct(
        r.classes,
        point[..r.n].iter().map(|&x| x as usize).collect(),
        point[r.n..2 * r.n].iter().map(|&x| x as usize).collect(),
        decoder,
    )
    .unwrap()
    .output_labels
}
#[test]
fn nonunique_prime_power_banks_recombine_to_actual_composite_tables() {
    for q in [4usize, 6, 12, 24, 30, 40] {
        let r = fixture(q);
        let out = checked(&r);
        assert!(!out.candidates.is_empty(), "q={q}");
        if q == 24 || q == 40 {
            assert!(out.candidates.len() >= 2);
        }
        let target: Vec<_> = (0..q * q)
            .map(|cell| (2 * (cell % q) + cell / q) % q)
            .collect();
        for &candidate in &out.candidates {
            assert_eq!(predictions(&r, &out.branches[candidate].point), target);
        }
        assert_eq!(out.work.ce_gradients, 0);
        if q == 4 {
            assert_eq!(out.work.crt_steps, 0);
        } else {
            assert!(out.work.crt_steps > 0);
        }
        if q == 30 {
            assert_eq!(out.components.len(), 3);
        }
    }
}
#[test]
fn missing_row_produces_distinct_completions_even_after_crt() {
    let mut r = fixture(6);
    r.system.rows.retain(|row| row[5] == 0);
    r.system.right = vec![0; r.system.rows.len()];
    let out = checked(&r);
    let tables: std::collections::BTreeSet<_> = out
        .candidates
        .iter()
        .map(|&i| predictions(&r, &out.branches[i].point))
        .collect();
    let missing_answers: std::collections::BTreeSet<_> =
        tables.iter().map(|table| table[5]).collect();
    assert_eq!(missing_answers, (0..6).collect());
    for table in tables {
        for j in 0..6 {
            for i in 0..5 {
                assert_eq!(table[i + 6 * j], (2 * i + j) % 6);
            }
        }
    }
}
#[test]
fn balanced_component_marginals_do_not_replace_joint_code_injectivity() {
    // Both residue marginals are uniform, but the actual joint code repeats
    // zero and five. Numeric constants define a negative fixture only.
    let point = [0, 0, 0, 0, 0, 0, 4, 1, 5, 5];
    let rows: Vec<_> = (0..point.len())
        .filter(|&i| i != 7)
        .map(|i| {
            let mut row = vec![0; point.len()];
            row[i] = 1;
            row[7] -= point[i];
            row
        })
        .collect();
    let r = Request {
        n: 2,
        classes: 6,
        system: System {
            variables: point.len(),
            right: vec![0; rows.len()],
            rows,
        },
        component_policy: component::Policy::default(),
        policy: Policy::default(),
    };
    let out = checked(&r);
    assert!(out.components.iter().all(|c| !c.complete.is_empty()));
    assert_eq!(out.branches.len(), 1);
    assert_eq!(out.branches[0].collision, Some([0, 1, 0]));
    assert!(out.candidates.is_empty());
}
#[test]
fn actual_tuple_prefixes_and_refused_components_keep_their_boundaries() {
    let r = fixture(24);
    let full = checked(&r);
    assert!(full.entire_emitted_product);
    for cap in [0, 1] {
        let mut r = r.clone();
        r.policy.points = cap;
        let out = checked(&r);
        assert_eq!(out.branches, full.branches[..cap]);
        assert_eq!(out.reason, "point_cap");
        assert!(!out.entire_emitted_product);
    }
    let cost = r.products_per_branch(full.components.len());
    for cap in [0, cost - 1, cost, cost + 1] {
        let mut r = r.clone();
        r.policy.original_products = cap;
        let out = checked(&r);
        let len = (cap / cost) as usize;
        assert_eq!(out.branches, full.branches[..len]);
        assert_eq!(out.reason, "original_product_cap");
    }
    let mut r = fixture(6);
    r.component_policy.enumeration.points = 1;
    let out = checked(&r);
    assert_eq!(out.components.len(), 2);
    assert_eq!(out.cardinality, 0);
    assert_eq!(out.reason, "component_refused");
    assert!(out.entire_emitted_product);
    assert!(out.branches.is_empty());
}
#[test]
fn changed_sources_witnesses_tuples_collisions_and_work_are_rejected() {
    let r = fixture(24);
    let out = checked(&r);
    for slot in 0..16 {
        let mut bad = out.clone();
        match slot {
            0 => bad.components[0].input.system.rows[0][0] += 1,
            1 => bad.branches[0].component_states[0] = usize::MAX,
            2 => bad.branches[0].crt[0].bezout = Some([0, 0]),
            3 => bad.branches[0].crt[0].input.left[0] = 1,
            4 => bad.branches[0].point[0] = 1,
            5 => bad.branches[0].collision = Some([0, 1, 0]),
            6 => bad.branches[0].duplicate = Some(0),
            7 => bad.branches[0].retained = false,
            8 => bad.branches[0].ordinal = 1,
            9 => bad.work.crt_original_products -= 1,
            10 => bad.work.endpoint_original_products -= 1,
            11 => bad.work.class_coordinates -= 1,
            12 => bad.work.ce_gradients = 1,
            13 => bad.cardinality += 1,
            14 => bad.candidates.clear(),
            _ => {
                bad.branches.pop();
            }
        }
        assert!(check(&r, &bad).is_err(), "mutation {slot}");
    }
    let mut invalid = r;
    invalid.classes = 0;
    assert!(run(invalid).is_err());
}

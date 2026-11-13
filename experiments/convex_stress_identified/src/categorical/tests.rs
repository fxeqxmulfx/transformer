use super::*;

fn fixture(q: usize) -> Request {
    let n = q + 1;
    let mut cells: Vec<_> = (0..q).map(|j| [0, j]).collect();
    cells.extend((0..q - 1).map(|j| [1, j]));
    cells.extend((2..n).map(|i| [i, 0]));
    cells.push([0, q]);
    let names: Vec<_> = (0..q).map(|c| 17 + q - 1 - c).collect();
    Request {
        data: Data {
            n,
            vocabulary: 2 * q + 20,
            observations: cells
                .iter()
                .map(|&[i, j]| [i + n * j, names[(i + j) % q]])
                .collect(),
        },
        proposal: Proposal {
            rows: (0..n).map(|i| ((i + 2) % q) as u16).collect(),
            columns: (0..n).map(|j| ((j + q - 1) % q) as u16).collect(),
            codes: (0..q).map(|c| ((c + 1) % q) as u16).collect(),
        },
        retained_equations: 4096,
        policy: ring::Policy::default(),
    }
}
fn predict(r: &Request, classes: &[usize], i: usize, j: usize) -> usize {
    let q = classes.len();
    let v = (r.proposal.rows[i] as usize + r.proposal.columns[j] as usize) % q;
    classes[r
        .proposal
        .codes
        .iter()
        .position(|&x| x as usize == v)
        .unwrap()]
}

#[test]
fn masked_domains_2_to_16_with_repeats_and_unknown_shifted_codes() {
    for q in 2..=16 {
        let r = fixture(q);
        let out = run(r.clone()).unwrap();
        assert!(out.certified, "q={q}: {}", out.reason);
        check(&r, &out).unwrap();
        replay(&r, &out).unwrap();
        let mut unseen = 0;
        for i in 0..r.data.n {
            for j in 0..r.data.n {
                if !r.data.observations.iter().any(|s| s[0] == i + r.data.n * j) {
                    unseen += 1;
                }
                assert_eq!(predict(&r, &out.classes, i, j), 17 + q - 1 - (i + j) % q);
            }
        }
        assert!(unseen > 0);
        assert_eq!(out.work.observation_checks, r.data.observations.len());
        assert_eq!(out.normalized.as_ref().unwrap()[out.anchor_row], 0);
        assert_eq!(out.work.ce_gradients, 0);
    }
}

#[test]
fn exact_six_state_lean_witness_and_categorical_renaming() {
    let cells = [
        [0, 0],
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [0, 5],
        [1, 0],
        [1, 1],
        [1, 2],
        [1, 3],
        [1, 4],
        [2, 0],
        [3, 0],
        [4, 0],
        [5, 0],
        [6, 0],
        [0, 6],
    ];
    let labels = [1, 0, 2, 3, 4, 5, 0, 2, 3, 4, 5, 2, 3, 4, 5, 1, 1];
    let mut r = fixture(6);
    r.data.vocabulary = 1024;
    r.data.observations = cells
        .iter()
        .zip(labels)
        .map(|(&[i, j], c)| [i + 7 * j, c])
        .collect();
    r.proposal = Proposal {
        rows: vec![0, 1, 2, 3, 4, 5, 0],
        columns: vec![0, 1, 2, 3, 4, 5, 0],
        codes: vec![0, 1, 2, 3, 4, 5],
    };
    let out = run(r.clone()).unwrap();
    assert!(out.certified);
    check(&r, &out).unwrap();
    replay(&r, &out).unwrap();
    assert_eq!(predict(&r, &out.classes, 2, 3), 5);
    assert_eq!(predict(&r, &out.classes, 2, 4), 1);
    let renaming = [615, 99, 29, 812, 506, 13];
    for s in &mut r.data.observations {
        s[1] = renaming[s[1]];
    }
    let renamed = run(r.clone()).unwrap();
    assert!(renamed.certified);
    check(&r, &renamed).unwrap();
    replay(&r, &renamed).unwrap();
    assert_eq!(out.kernel, renamed.kernel);
    assert_eq!(predict(&r, &renamed.classes, 2, 3), renaming[5]);
    assert_eq!(predict(&r, &renamed.classes, 2, 4), renaming[1]);
}

#[test]
fn missing_axes_fit_but_do_not_receive_a_certificate() {
    for missing_rows in [true, false] {
        let mut r = fixture(6);
        r.data.observations = (0..7)
            .map(|k| {
                let (i, j) = if missing_rows { (0, k) } else { (k, 0) };
                [i + 7 * j, 22 - (i + j) % 6]
            })
            .collect();
        let out = run(r.clone()).unwrap();
        assert_eq!(out.reason, "missing_axis_refusal");
        assert!(!out.certified);
        assert_eq!(out.work.observation_checks, 7);
        check(&r, &out).unwrap();
        replay(&r, &out).unwrap();
        let before = predict(&r, &out.classes, 1, 1);
        if missing_rows {
            r.proposal.rows[1] = (r.proposal.rows[1] + 1) % 6;
        } else {
            r.proposal.columns[1] = (r.proposal.columns[1] + 1) % 6;
        }
        let alternate = run(r.clone()).unwrap();
        assert_eq!(alternate.reason, "missing_axis_refusal");
        assert_ne!(before, predict(&r, &alternate.classes, 1, 1));
        check(&r, &alternate).unwrap();
        replay(&r, &alternate).unwrap();
    }
}

#[test]
fn code_conflicts_equation_caps_and_larger_kernels_are_retained() {
    let mut r = fixture(6);
    r.proposal.codes[1] = r.proposal.codes[0];
    let out = run(r.clone()).unwrap();
    assert_eq!(out.reason, "nonbijective_code");
    check(&r, &out).unwrap();
    replay(&r, &out).unwrap();
    let mut r = fixture(6);
    r.retained_equations = 5;
    let out = run(r.clone()).unwrap();
    assert_eq!(out.reason, "kernel_insufficient_rows");
    check(&r, &out).unwrap();
    replay(&r, &out).unwrap();
    // Even an equation beyond the elimination cap is checked against the proposal.
    r.data.observations.last_mut().unwrap()[1] = 21;
    let out = run(r.clone()).unwrap();
    assert_eq!(out.reason, "observed_equation_conflict");
    assert_eq!(out.conflict, Some(16));
    check(&r, &out).unwrap();
    replay(&r, &out).unwrap();
    let mut r = fixture(6);
    r.policy.scalar_updates = 0;
    let out = run(r.clone()).unwrap();
    assert_eq!(out.reason, "kernel_scalar_update_cap");
    check(&r, &out).unwrap();
    replay(&r, &out).unwrap();

    // A valid cyclic model can have a noncyclic observed joint module. Refusal
    // is not a proof of impossible decoded recovery in a richer program.
    let r = Request {
        data: Data {
            n: 4,
            vocabulary: 4,
            observations: (0..4)
                .flat_map(|j| {
                    (0..4)
                        .filter(move |&i| (i, j) != (3, 3))
                        .map(move |i| [i + 4 * j, (2 * i + j) % 4])
                })
                .collect(),
        },
        proposal: Proposal {
            rows: vec![0, 2, 0, 2],
            columns: vec![0, 1, 2, 3],
            codes: vec![0, 2, 1, 3],
        },
        retained_equations: 4096,
        policy: ring::Policy::default(),
    };
    let out = run(r.clone()).unwrap();
    assert!(!out.certified);
    assert_eq!(out.reason, "kernel_nonunit_remaining_column");
    check(&r, &out).unwrap();
    replay(&r, &out).unwrap();
}

#[test]
fn rehashed_original_matrix_and_endpoint_mutations_are_rejected() {
    let r = fixture(6);
    let original = run(r.clone()).unwrap();
    for slot in 0..11 {
        let mut out = original.clone();
        match slot {
            0 => out.classes.swap(0, 1),
            1 => out.anchor_row = 1,
            2 => out.anchor_class = Some(0),
            3 => out.normalized.as_mut().unwrap()[0] = 1,
            4 => out.missing_rows.push(0),
            5 => out.work.observation_checks -= 1,
            6 => out.work.normalized_coordinates += 1,
            7 => out.collision = Some([0, 1]),
            8 => out.reason = "unconditional_ground_truth".into(),
            9 => out.kernel.as_mut().unwrap().input.rows[0][0] = 0,
            _ => {
                let k = out.kernel.as_mut().unwrap();
                k.input.rows.push(vec![0; k.input.variables]);
            }
        }
        assert!(check(&r, &out).is_err(), "mutation{slot}");
        assert!(replay(&r, &out).is_err());
    }
    let mut out = original;
    out.kernel.as_mut().unwrap().work.gcd_steps += 1;
    assert!(replay(&r, &out).is_err());
}

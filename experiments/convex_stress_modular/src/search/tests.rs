use super::*;

fn fixture(q: usize) -> (Input, Vec<usize>) {
    let n = q + 2;
    // Unequal active axis counts exercise padding. One entire column and two
    // rows have the same categorical label, independent of the interior code.
    let labels: Vec<_> = (0..n * n)
        .map(|cell| {
            let (i, j) = (cell % n, cell / n);
            if i < 2 || j == n - 1 {
                0
            } else {
                2 * ((i - 2 + q - 1 + j) % q) + 1
            }
        })
        .collect();
    let observations = (0..n * n)
        .filter(|&cell| cell != 3 * n + 3 && cell != 2 * n + 4)
        .map(|cell| [cell, labels[cell]])
        .collect();
    (
        Input {
            categorical: coding::Input {
                n,
                vocab: 2 * q + 4,
                observations,
                policy: coding::Policy::default(),
            },
            minimum_strip_support: 3,
            max_views: 4,
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
fn composite_interior_with_categorical_strips_padding_and_renaming_recovers() {
    for q in [6, 8, 12] {
        let (input, labels) = fixture(q);
        let out = checked(&input);
        assert_eq!(out.reason, "admitted_proposals");
        let selected = &out.candidates[out.selected.unwrap()];
        assert_eq!(selected.model.output_labels, labels);
        assert_eq!(selected.model.q, 3 * q);
        assert_eq!(out.attempts[selected.view].view.row_padding, 1);
        assert_eq!(selected.audit["training_errors"], 0);
        let mut renamed = input.clone();
        for s in &mut renamed.categorical.observations {
            s[1] = input.categorical.vocab - 1 - s[1];
        }
        let other = checked(&renamed);
        let model = &other.candidates[other.selected.unwrap()].model;
        assert_eq!(model.rows, selected.model.rows);
        assert_eq!(model.columns, selected.model.columns);
        assert!(
            model
                .output_labels
                .iter()
                .zip(labels)
                .all(|(&a, b)| a + b == input.categorical.vocab - 1)
        );
    }
}

#[test]
fn view_caps_small_interiors_and_component_failures_remain_recorded() {
    let (mut input, _) = fixture(6);
    let complete = checked(&input);
    input.max_views = 0;
    let out = checked(&input);
    assert_eq!(out.attempts.len(), complete.attempts.len());
    assert!(out.attempts.iter().all(|a| a.decision == Decision::ViewCap));
    input.max_views = 1;
    let out = checked(&input);
    assert_eq!(out.attempts[1].decision, Decision::ViewCap);
    assert!(out.candidates.is_empty());
    input.max_views = 4;
    input.categorical.policy.stages = 0;
    assert!(
        checked(&input)
            .attempts
            .iter()
            .all(|a| a.decision == Decision::Refused)
    );
    input.categorical.policy = coding::Policy::default();
    for s in &mut input.categorical.observations {
        s[1] = 0;
    }
    let out = checked(&input);
    assert_eq!(out.attempts[1].decision, Decision::EmptyInterior);
    assert!(out.candidates.is_empty());
}

#[test]
fn changed_view_component_full_table_audit_selection_and_work_are_rejected() {
    let (input, _) = fixture(6);
    let out = checked(&input);
    let selected = out.candidates[0].view;
    for mutation in 0..9 {
        let mut bad = out.clone();
        match mutation {
            0 => bad.attempts[selected].view.rows[0] = Some(0),
            1 => bad.attempts[selected].view.row_support[0] += 1,
            2 => {
                bad.attempts[selected]
                    .coding
                    .as_mut()
                    .unwrap()
                    .point
                    .as_mut()
                    .unwrap()[0] += 1
            }
            3 => bad.candidates[0].model.output_labels[0] += 1,
            4 => bad.candidates[0].audit["train_ce"] = 0.into(),
            5 => bad.selected = None,
            6 => bad.work.full_raw_frequencies += 1,
            7 => bad.work.ce_gradients = 1,
            _ => {
                bad.attempts.pop();
            }
        }
        assert!(check(&input, &bad).is_err(), "mutation={mutation}");
    }
    let mut input = input;
    input.max_views = 5;
    assert!(run(input).is_err());
}

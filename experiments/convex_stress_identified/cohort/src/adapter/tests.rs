use super::*;
use convex_stress_modular::coding;
use serde_json::json;

pub(crate) fn fixture() -> search::Report {
    let n = 8;
    let input = search::Input {
        categorical: coding::Input {
            n,
            vocab: 32,
            observations: (0..n * n)
                .filter(|&cell| cell != 2 + n * 3 && cell != 7 + n * 7)
                .map(|cell| {
                    let (i, j) = (cell % n, cell / n);
                    [
                        cell,
                        if i == 0 || j == 0 {
                            31
                        } else {
                            13 + 2 * ((i + j - 2) % 6)
                        },
                    ]
                })
                .collect(),
            policy: coding::Policy::default(),
        },
        minimum_strip_support: 8,
        max_views: 4,
    };
    let parent = search::run(input).unwrap();
    search::check(&parent.input, &parent).unwrap();
    parent
}
fn checked(parent: &search::Report, policy: Policy) -> Report {
    let out = run(parent, policy.clone()).unwrap();
    check(parent, &policy, &out).unwrap();
    replay(parent, &out).unwrap();
    assert_eq!(
        serde_json::from_slice::<Report>(&serde_json::to_vec(&out).unwrap()).unwrap(),
        out
    );
    out
}

#[test]
fn original_proposal_and_fixed_strip_scope_are_bound_to_actual_certificates() {
    let parent = fixture();
    let out = checked(&parent, Policy::default());
    assert_eq!(parent.candidates.len(), 1);
    assert_eq!(out.certificate_count, 1);
    let e = &out.entries[0];
    assert_eq!(e.full_prediction_checks, 64);
    assert!(e.scope.unknown_bijective_interior_required);
    assert!(e.scope.constant_strip_structure_required);
    assert_eq!(e.scope.constant, Some(31));
    assert!(!e.scope.unconditional_ground_truth_identification);
    assert_eq!(e.scope.row_map[0], None);
    assert_eq!(e.scope.column_map[0], None);
    assert_eq!(out.parent_views.len(), parent.attempts.len());
    for cell in 0..64 {
        let (i, j) = (cell % 8, cell / 8);
        assert_eq!(
            parent.candidates[0].model.output_labels[cell],
            if i == 0 || j == 0 {
                31
            } else {
                13 + 2 * ((i + j - 2) % 6)
            }
        );
    }
    assert_eq!(out.ce_gradients, 0);
}

#[test]
fn earlier_missing_axis_pins_do_not_become_identification_certificates() {
    let n = 7;
    let parent = search::run(search::Input {
        categorical: coding::Input {
            n,
            vocab: 8,
            observations: (0..n * n)
                .filter(|&cell| cell % n != 5)
                .map(|cell| [cell, (cell % n + cell / n) % 6])
                .collect(),
            policy: coding::Policy::default(),
        },
        minimum_strip_support: 8,
        max_views: 4,
    })
    .unwrap();
    assert_eq!(parent.candidates.len(), 1);
    let out = checked(&parent, Policy::default());
    assert_eq!(out.certificate_count, 0);
    assert_eq!(out.entries[0].certificate.reason, "missing_axis_refusal");
    assert_eq!(out.entries[0].certificate.missing_rows, vec![5]);
    assert!(out.entries[0].certificate.kernel.is_none());
}

#[test]
fn ce_ties_use_only_checked_structure_and_not_original_split_scores() {
    let parent = fixture();
    let out = checked(&parent, Policy::default());
    let ce = parent.candidates[0].audit["train_ce"].as_f64().unwrap();
    let mut choices = vec![json!({"training_mode":"full_batch","selected":{
        "train_ce":ce,"model_sha256":"different-exact-fit","metrics":{"unseen":{"sequence_accuracy":1.0}},
        "primary_pass":true,"all_splits_pass":true}})];
    let reference = select(&parent, &out, &choices).unwrap();
    assert_eq!(reference[0]["selected"], "identified-0");
    choices[0]["selected"]["metrics"] = json!({"unseen":{"sequence_accuracy":0.0}});
    choices[0]["selected"]["primary_pass"] = false.into();
    choices[0]["selected"]["all_splits_pass"] = false.into();
    choices[0]["selection"] = json!("changed-primary-name");
    assert_eq!(select(&parent, &out, &choices).unwrap(), reference);
    choices[0]["selected"]["train_ce"] = json!(ce - 1e-8);
    assert_eq!(
        select(&parent, &out, &choices).unwrap()[0]["selected"],
        "parent"
    );
    choices[0]["selected"]["train_ce"] = json!(ce + 1e-8);
    assert_eq!(
        select(&parent, &out, &choices).unwrap()[0]["selected"],
        "identified-0"
    );
    choices[0]["selected"]["train_ce"] = json!(ce);
    choices[0]["selected"]["model_sha256"] = out.entries[0].model_sha256.clone().into();
    assert_eq!(
        select(&parent, &out, &choices).unwrap()[0]["selected"],
        "parent"
    );
    choices[0]["selected"] = Value::Null;
    assert_eq!(
        select(&parent, &out, &choices).unwrap()[0]["selected"],
        "identified-0"
    );
    choices[0]["selected"] = json!({"train_ce":"NaN"});
    assert!(select(&parent, &out, &choices).is_err());
}

#[test]
fn rehashed_model_scope_matrix_and_parent_mutations_are_rejected() {
    let parent = fixture();
    let out = checked(&parent, Policy::default());
    for slot in 0..10 {
        let mut bad = out.clone();
        match slot {
            0 => {
                bad.entries[0]
                    .scope
                    .unconditional_ground_truth_identification = true
            }
            1 => bad.entries[0].scope.row_map[0] = Some(0),
            2 => bad.entries[0].model_sha256 = "different".into(),
            3 => bad.entries[0].request_sha256 = "different".into(),
            4 => bad.entries[0].full_prediction_checks -= 1,
            5 => {
                bad.entries[0]
                    .certificate
                    .kernel
                    .as_mut()
                    .unwrap()
                    .input
                    .rows[0][0] = 0
            }
            6 => bad.entries[0].certificate.normalized.as_mut().unwrap()[0] = 1,
            7 => bad.certificate_count = 0,
            8 => {
                bad.parent_views.pop();
            }
            _ => bad.ce_gradients = 1,
        }
        assert!(check(&parent, &out.policy, &bad).is_err(), "mutation{slot}");
    }
    for slot in 0..4 {
        let mut bad = parent.clone();
        match slot {
            0 => bad.candidates[0].model.output_labels[0] = 0,
            1 => bad.candidates[0].audit["train_ce"] = json!(0.0),
            2 => {
                bad.attempts[bad.candidates[0].view]
                    .coding
                    .as_mut()
                    .unwrap()
                    .point
                    .as_mut()
                    .unwrap()[0] += 1
            }
            _ => bad.attempts[bad.candidates[0].view].decision = search::Decision::Refused,
        }
        assert!(check(&bad, &out.policy, &out).is_err(), "parent{slot}");
    }
}

#[test]
fn empty_banks_caps_and_refused_certificates_keep_the_earlier_choice() {
    let parent = fixture();
    let mut p = Policy::default();
    p.ring.scalar_updates = 0;
    let out = checked(&parent, p);
    assert_eq!(out.certificate_count, 0);
    assert_eq!(
        out.entries[0].certificate.reason,
        "kernel_scalar_update_cap"
    );
    let choices = vec![json!({"training_mode":"forward_only","selected":{
        "train_ce":parent.candidates[0].audit["train_ce"],"model_sha256":"earlier"}})];
    assert_eq!(
        select(&parent, &out, &choices).unwrap()[0]["selected"],
        "parent"
    );
    let mut input = parent.input;
    input.max_views = 0;
    let parent = search::run(input).unwrap();
    let out = checked(&parent, Policy::default());
    assert!(out.entries.is_empty());
    assert_eq!(out.parent_views.len(), parent.attempts.len());
    assert!(out.parent_views.iter().all(|v| v["decision"] == "ViewCap"));
    assert_eq!(
        select(&parent, &out, &choices).unwrap()[0]["selected"],
        "parent"
    );
    let mut invalid = Policy::default();
    invalid.ring.scalar_updates = u64::MAX;
    assert!(run(&parent, invalid).is_err());
    let invalid = Policy {
        retained_equations: 0,
        ..Policy::default()
    };
    assert!(run(&parent, invalid).is_err());
}

use super::*;
use serde_json::json;

fn fixture() -> (Input, Vec<usize>) {
    let n = 5;
    let target: Vec<_> = (0..n * n)
        .map(|cell| {
            let (r, c) = (cell % n, cell / n);
            if r == 4 || c == 4 {
                0
            } else {
                (2 * r + c + 1) % 4 + 3
            }
        })
        .collect();
    let observations = target
        .iter()
        .enumerate()
        .filter(|&(cell, _)| cell != 18)
        .map(|(cell, &label)| [cell, label])
        .collect();
    (
        Input {
            categorical: learner::Request {
                n,
                vocabulary: 11,
                observations,
                policy: learner::Policy::default(),
            },
            minimum_strip_support: 2,
            max_views: 4,
        },
        target,
    )
}
fn checked(input: &Input, view: usize) -> Report {
    let request = Request {
        input: input.clone(),
        view,
    };
    let out = run(request.clone()).unwrap();
    check(&request, &out).unwrap();
    replay(&request, &out).unwrap();
    assert_eq!(
        serde_json::from_slice::<Report>(&serde_json::to_vec(&out).unwrap()).unwrap(),
        out
    );
    out
}
#[test]
fn fixed_view_binds_outside_strip_label_raw_moments_and_actual_query_programs() {
    let (input, target) = fixture();
    let slots = catalogue(&input).unwrap();
    assert_eq!(slots.len(), 2);
    for (index, slot) in slots.iter().enumerate() {
        let out = checked(&input, index);
        assert_eq!(&out.slot, slot);
        if slot.view.constant.is_none() {
            continue;
        }
        let c = &out.candidates[out.selected.unwrap()];
        assert_eq!(c.model.output_labels, target);
        assert_eq!(c.model.q, 12);
        assert_eq!(c.scope.constant, Some(0));
        assert!(!c.scope.class_to_output.contains(&0));
        assert_eq!(c.scope.certified_interior_cells, 16);
        assert_eq!(c.scope.assumed_strip_cells, 9);
        assert_eq!(c.scope.uncertified_cells, 0);
        assert!(c.scope.full_table_certified_within_family);
        assert!(!c.scope.unconditional_identification);
        assert!(matches!(
            c.scope.origins[18],
            Origin::ObservedConsequence {
                interior_cell: 15,
                ..
            }
        ));
        assert_eq!(c.scope.origins[24], Origin::AssumedConstantStrip);
        assert_eq!(c.raw_audit["vocabulary"], 11);
        assert_eq!(c.raw_audit["training_errors"], 0);
        assert_eq!(
            c.raw_audit["latent_moment_norms"].as_array().unwrap().len(),
            12
        );
        assert_eq!(c.raw_audit["exact_real_arithmetic_certificate"], false);
        assert_eq!(out.work.ce_gradients, 0);
    }
}
#[test]
fn padded_unused_interior_does_not_identify_missing_strips_without_the_view_assumption() {
    let (mut input, target) = fixture();
    input.categorical.observations.retain(|s| s[0] % 5 != 3);
    let slots = catalogue(&input).unwrap();
    let index = slots
        .iter()
        .position(|s| s.view.constant == Some(0))
        .unwrap();
    assert_eq!(slots[index].view.missing_rows, vec![3]);
    assert_eq!(slots[index].view.row_padding, 1);
    let out = checked(&input, index);
    let inner = out.interior.as_ref().unwrap();
    assert!(inner.candidates.iter().all(|c| !c.certificate.certified));
    assert_eq!(out.candidates.len(), 2);
    assert_eq!(out.candidates[1].duplicate_of, Some(0));
    for c in &out.candidates {
        assert!(c.scope.full_table_certified_within_family);
        assert!(!c.scope.unconditional_identification);
        assert_eq!(c.scope.certified_interior_cells, 12);
        assert_eq!(c.scope.assumed_strip_cells, 13);
        assert_eq!(c.model.output_labels[3], 0);
        assert_ne!(c.model.output_labels[3], target[3]);
    }
    let base = checked(&input, 0);
    assert!(
        base.candidates
            .iter()
            .all(|c| !c.scope.full_table_certified_within_family)
    );
}
#[test]
fn catalogue_retains_caps_empty_views_preflight_refusals_and_invalid_policies() {
    let (mut input, _) = fixture();
    input.max_views = 0;
    for (index, slot) in catalogue(&input).unwrap().iter().enumerate() {
        assert_eq!(slot.disposition, Disposition::ViewCap);
        assert!(checked(&input, index).interior.is_none());
    }
    input.max_views = 4;
    for sample in &mut input.categorical.observations {
        sample[1] = 0;
    }
    let slots = catalogue(&input).unwrap();
    assert_eq!(slots[1].disposition, Disposition::EmptyInterior);
    assert!(checked(&input, 1).interior.is_none());
    assert!(checked(&input, 0).candidates.is_empty());
    input.categorical.policy.models = 9;
    assert!(catalogue(&input).is_err());
    let (mut input, _) = fixture();
    input.categorical.n = 257;
    input.categorical.vocabulary = 258;
    input.categorical.policy.models = 8;
    input.categorical.observations = (0..4096).map(|cell| [cell, cell % 210]).collect();
    let slots = catalogue(&input).unwrap();
    assert!(matches!(
        slots[0].disposition,
        Disposition::PreflightRefusal { .. }
    ));
    assert!(checked(&input, 0).interior.is_none());
}
#[test]
fn full_view_mutations_cannot_change_maps_scope_programs_raw_ce_or_work() {
    let (input, _) = fixture();
    let out = checked(&input, 1);
    for mutation in 0..15 {
        let mut bad = out.clone();
        match mutation {
            0 => bad.slot.view.rows[0] = None,
            1 => bad.candidates[0].model.decoder[1] = 1,
            2 => bad.candidates[0].model.output_labels[24] = 1,
            3 => bad.candidates[0].scope.class_to_output.swap(0, 1),
            4 => bad.candidates[0].scope.constant = Some(1),
            5 => bad.candidates[0].scope.unconditional_identification = true,
            6 => bad.candidates[0].scope.target_family.clear(),
            7 => bad.candidates[0].scope.origins[18] = Origin::AssumedConstantStrip,
            8 => bad.candidates[0].raw_audit["train_ce"] = json!(0.),
            9 => bad.candidates[0].raw_audit["latent_moment_norms"][0]["operator"] = json!(100.),
            10 => bad.work.completed_raw_frequencies -= 1,
            11 => bad.candidates[0].duplicate_of = Some(0),
            12 => {
                bad.interior.as_mut().unwrap().candidates[0]
                    .certificate
                    .queries[15]
                    .label = 0
            }
            13 => bad.decisions.clear(),
            _ => bad.selected = None,
        }
        assert!(check(&out.input, &bad).is_err(), "mutation {mutation}");
    }
}
#[test]
fn full_view_selection_reads_only_observed_ce_and_conditional_certificate_priority() {
    let (input, _) = fixture();
    let c = checked(&input, 1).candidates.remove(0);
    let mut bank = vec![c.clone(), c];
    bank[0].scope.full_table_certified_within_family = false;
    bank[0].raw_audit["held_out_accuracy"] = json!(1.);
    bank[1].raw_audit["held_out_accuracy"] = json!(0.);
    assert_eq!(plan::select(&bank).unwrap(), Some(1));
    bank[0].raw_audit["train_ce"] = json!(0.);
    assert_eq!(plan::select(&bank).unwrap(), Some(0));
    bank[0] = bank[1].clone();
    assert_eq!(plan::select(&bank).unwrap(), Some(0));
}

use super::*;
use serde_json::json;

fn fixture(q: usize) -> (Request, Vec<usize>) {
    let target: Vec<_> = (0..q * q)
        .map(|cell| (2 * (cell % q) + cell / q + 1) % q + 3)
        .collect();
    let observations = target
        .iter()
        .enumerate()
        .filter(|&(cell, _)| cell != q * q - 1)
        .map(|(cell, &label)| [cell, label])
        .collect();
    (
        Request {
            n: q,
            vocabulary: q + 7,
            observations,
            policy: Policy::default(),
        },
        target,
    )
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
fn observed_only_discovery_binds_actual_complex_models_ce_and_decoded_certificates() {
    for q in [4usize, 8, 12, 17] {
        let (r, target) = fixture(q);
        let out = checked(&r);
        assert!(out.selected.is_some(), "q={q}");
        for candidate in &out.candidates {
            assert_eq!(candidate.model.output_labels, target);
            assert!(candidate.certificate.certified);
            assert_eq!(
                candidate.certificate.input.predictions,
                candidate.model.output_labels
            );
            assert_eq!(candidate.raw_audit["training_errors"], 0);
            assert_eq!(candidate.raw_audit["vocabulary"], r.vocabulary);
            assert_eq!(
                candidate.raw_audit["exact_real_arithmetic_certificate"],
                false
            );
            assert_eq!(
                candidate.raw_audit["latent_moment_norms"]
                    .as_array()
                    .unwrap()
                    .len(),
                q
            );
        }
        if q == 8 {
            assert!(
                out.attempts
                    .iter()
                    .any(|a| matches!(a.decision, Decision::Duplicate { .. }))
            );
        }
        assert!(out.work.raw_frequency_extents.iter().all(Option::is_some));
        assert_eq!(out.work.ce_gradients, 0);
    }
}

#[test]
fn ninety_six_state_half_observed_noncyclic_model_is_discovered_and_certified() {
    large_half_observed(96);
}

#[test]
fn one_ninety_two_state_half_observed_model_fits_the_same_certificate_budgets() {
    large_half_observed(192);
}

fn large_half_observed(n: usize) {
    let (mut r, target) = fixture(n);
    // A fixed permutation chooses and orders observed cells. It is fixture
    // data selection, not a feature supplied to the learner.
    r.observations.sort_by_key(|sample| {
        let mut x = (sample[0] as u64).wrapping_add(0x9e3779b97f4a7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^ (x >> 31)
    });
    r.observations.truncate(n * n / 2);
    let out = checked(&r);
    let selected = out.selected.expect("large composite candidate missing");
    let candidate = &out.candidates[selected];
    assert_eq!(candidate.model.output_labels, target);
    assert!(candidate.certificate.certified);
    assert_eq!(candidate.certificate.queries.len(), n * n);
    assert_eq!(candidate.raw_audit["training_errors"], 0);
    let code = out.construction.as_ref().unwrap();
    eprintln!(
        "{}",
        json!({"synthetic_n":n,"observations":r.observations.len(),
        "completed_code_tuples":code.candidates.len(),"models":out.candidates.len(),
        "component_stages":code.components.iter().map(|c|c.work.stages).sum::<usize>(),
        "component_scalar_updates":code.components.iter().map(|c|c.work.finite_scalar_updates).sum::<u64>(),
        "certificate_prefix_updates":candidate.certificate.prefix.work.scalar_updates,
        "query_reduction_products":candidate.certificate.work.reduction_products,
        "query_certificate_products":candidate.certificate.work.certificate_products,
        "certified_queries":candidate.certificate.queries.len(),"ce_gradients":out.work.ce_gradients})
    );
}
#[test]
fn arbitrary_class_names_and_unused_vocabulary_preserve_original_decoding() {
    let (mut r, mut target) = fixture(8);
    let names = [7, 0, 5, 2, 6, 1, 4, 3];
    for label in &mut target {
        *label = 20 + names[*label - 3];
    }
    for sample in &mut r.observations {
        sample[1] = 20 + names[sample[1] - 3];
    }
    r.vocabulary = 32;
    let out = checked(&r);
    assert!(!out.candidates.is_empty());
    for c in &out.candidates {
        assert_eq!(c.model.output_labels, target);
        assert!(c.certificate.certified);
        let expected = (std::f64::consts::E + 31.).ln() - 1.;
        assert!((c.raw_audit["train_ce"].as_f64().unwrap() - expected).abs() < 1e-12);
    }
}
#[test]
fn missing_axes_remain_free_and_ambiguous_models_remain_uncertified() {
    let (mut r, _) = fixture(4);
    r.observations.retain(|s| s[0] % 4 != 3);
    let out = checked(&r);
    assert_eq!(out.partition.missing_rows, vec![3]);
    let construction = out.construction.as_ref().unwrap();
    assert_eq!(
        construction.input.system.rows.len(),
        r.observations.len() + 2
    );
    assert!(construction.input.system.rows.iter().all(|row| row[3] == 0));
    assert_eq!(out.candidates.len(), 2);
    assert_ne!(
        out.candidates[0].model.output_labels,
        out.candidates[1].model.output_labels
    );
    assert!(out.candidates.iter().all(|c| !c.certificate.certified));
    assert!(
        out.attempts
            .iter()
            .any(|a| a.decision == Decision::ModelCap)
    );
}
#[test]
fn omitted_observation_conflicts_caps_and_unsupported_domains_are_retained() {
    let (mut r, _) = fixture(4);
    r.policy.equations = r.observations.len() - 1;
    let last = r.observations.last_mut().unwrap();
    last[1] = 3 + (last[1] - 3 + 1) % 4;
    let out = checked(&r);
    assert!(out.candidates.is_empty());
    assert!(
        out.attempts
            .iter()
            .any(|a| matches!(a.decision, Decision::ObservationConflict { .. }))
    );
    let (mut r, _) = fixture(4);
    r.policy.models = 0;
    let out = checked(&r);
    assert!(out.candidates.is_empty());
    assert!(
        out.attempts
            .iter()
            .any(|a| a.decision == Decision::ModelCap)
    );
    assert_eq!(out.work.raw_audits, 0);
    for sample in &mut r.observations {
        sample[1] = 0;
    }
    let out = checked(&r);
    assert_eq!(out.reason, "unsupported_latent_domain");
    r.policy.models = 9;
    assert!(run(r).is_err());
}
#[test]
fn changed_codes_models_spectra_ce_certificates_or_work_are_rejected() {
    let (r, _) = fixture(8);
    let out = checked(&r);
    for slot in 0..13 {
        let mut bad = out.clone();
        match slot {
            0 => bad.partition.classes.swap(0, 1),
            1 => {
                bad.candidates[0].model.latent_labels[0] =
                    (bad.candidates[0].model.latent_labels[0] + 1) % 8
            }
            2 => bad.candidates[0].model.output_labels[0] = 0,
            3 => bad.candidates[0].model.decoder[0] = 0,
            4 => bad.candidates[0].raw_audit["train_ce"] = json!(123.),
            5 => bad.candidates[0].raw_audit["latent_moment_norms"][0]["operator"] = json!(123.),
            6 => bad.candidates[0].certificate.input.predictions[0] = 0,
            7 => bad.candidates[0].certificate.queries[0].label = 0,
            8 => bad.attempts[0].branch = usize::MAX,
            9 => bad.selected = None,
            10 => bad.work.observation_checks -= 1,
            11 => bad.work.completed_raw_frequencies -= 1,
            _ => bad.construction.as_mut().unwrap().input.system.rows[0][0] += 1,
        }
        assert!(check(&r, &bad).is_err(), "mutation {slot}");
    }
}
#[test]
fn selection_uses_observed_ce_then_certification_then_original_order() {
    let (r, _) = fixture(4);
    let c = checked(&r).candidates.remove(0);
    let mut bank = vec![c.clone(), c];
    bank[0].certificate.certified = false;
    assert_eq!(plan::select(&bank).unwrap(), Some(1));
    bank[0].raw_audit["held_out_accuracy"] = json!(1.);
    bank[1].raw_audit["held_out_accuracy"] = json!(0.);
    assert_eq!(plan::select(&bank).unwrap(), Some(1));
    bank[0].certificate.certified = true;
    assert_eq!(plan::select(&bank).unwrap(), Some(0));
    bank[0].certificate.certified = false;
    bank[0].raw_audit["train_ce"] = json!(0.);
    assert_eq!(plan::select(&bank).unwrap(), Some(0));
}

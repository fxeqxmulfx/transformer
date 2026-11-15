use crate::{
    family::{Evaluator, programs},
    selection::{encode, select},
    *,
};
fn meter() -> Work {
    Work::new(joint::MAX_WORK).unwrap()
}
fn sample() -> Training {
    let mut rows = vec![];
    for a in 0..4u32 {
        for b in 0..7u32 {
            rows.push([a, b, (a + b).pow(2) % 7]);
        }
    }
    Training {
        vocabulary: 7,
        rows,
    }
}

#[test]
fn fixed_family_and_both_interpreters_on_complete_small_domain() {
    let family = programs();
    assert_eq!(family.len(), 4767);
    let unique: std::collections::BTreeSet<_> = family
        .iter()
        .map(|p| serde_json::to_string(p).unwrap())
        .collect();
    assert_eq!(unique.len(), family.len());
    let e = Evaluator::new(7).unwrap();
    for &p in &family {
        for a in 0..7 {
            for b in 0..7 {
                assert_eq!(e.evaluate(p, a, b), e.reference(p, a, b));
            }
        }
    }
    let parsed: serde_json::Value = serde_json::from_str(PROTOCOL).unwrap();
    assert_eq!(parsed["name"], FAMILY);
    assert_eq!(parsed["fixed_family"]["total"], PROGRAMS);
    for mode in [
        joint::training::Mode::Forward,
        joint::training::Mode::Full,
        joint::training::Mode::Mini,
    ] {
        let p = policy(mode);
        assert_eq!(
            p.initial_mass,
            parsed["training"]["initial_mass"].as_f64().unwrap()
        );
        assert_eq!(
            p.learning_rate,
            parsed["training"]["learning_rate"].as_f64().unwrap()
        );
        assert_eq!(
            p.batch_size,
            parsed["training"]["mini_batch_size"].as_u64().unwrap() as usize
        );
        let key = match mode {
            joint::training::Mode::Forward => "forward_updates",
            joint::training::Mode::Full => "full_epochs",
            joint::training::Mode::Mini => "mini_epochs",
        };
        assert_eq!(p.epochs, parsed["training"][key].as_u64().unwrap() as usize);
    }
}

#[test]
fn exact_elimination_matches_independent_pairwise_collision_definition() {
    let t = sample();
    let family = programs();
    let e = Evaluator::new(t.vocabulary).unwrap();
    let chosen = select(&t, &mut meter()).unwrap();
    let mut expected = vec![];
    for (id, &p) in family.iter().enumerate() {
        let states: Vec<_> = t
            .rows
            .iter()
            .map(|r| e.reference(p, r[0] as usize, r[1] as usize))
            .collect();
        let valid = (0..t.rows.len()).all(|i| {
            (0..t.rows.len()).all(|j| states[i] != states[j] || t.rows[i][2] == t.rows[j][2])
        });
        if valid {
            expected.push(id);
        }
    }
    assert!(!expected.is_empty());
    assert_eq!(chosen.retained, expected);
    assert_eq!(chosen.retained.len() + chosen.conflicts.len(), PROGRAMS);
    let mut renamed = t.clone();
    for row in &mut renamed.rows {
        row[2] = (2 * row[2] + 1) % 7;
    }
    assert_eq!(
        chosen.retained,
        select(&renamed, &mut meter()).unwrap().retained
    );
}

#[test]
fn all_compatible_candidates_survive_and_forged_subsets_are_rejected() {
    let t = Training {
        vocabulary: 7,
        rows: vec![[0, 1, 3], [0, 1, 3]],
    };
    let chosen = select(&t, &mut meter()).unwrap();
    assert_eq!(chosen.retained, (0..PROGRAMS).collect::<Vec<_>>());
    let (domain, encoded) = encode(&t, &chosen, &mut meter()).unwrap();
    assert_eq!(domain.shape.programs, PROGRAMS);
    assert_eq!(encoded.len(), 2);
    assert_eq!(encoded[0], encoded[1]);
    let mut forged = chosen.clone();
    forged.retained.pop();
    assert!(
        encode(&t, &forged, &mut meter())
            .unwrap_err()
            .to_string()
            .contains("omits")
    );
    let mut inconsistent = t;
    inconsistent.rows[1][2] = 4;
    let empty = select(&inconsistent, &mut meter()).unwrap();
    assert!(empty.retained.is_empty());
    assert_eq!(empty.conflicts.len(), PROGRAMS);
    assert!(encode(&inconsistent, &empty, &mut meter()).is_err());
}

#[test]
fn many_to_one_decoder_trains_all_modes_without_a_target_rule() {
    let t = sample();
    let mut work = meter();
    let selected = select(&t, &mut work).unwrap();
    let (domain, records) = encode(&t, &selected, &mut work).unwrap();
    let mut final_loss = vec![];
    for mode in [
        joint::training::Mode::Forward,
        joint::training::Mode::Full,
        joint::training::Mode::Mini,
    ] {
        let fit = joint::training::train(&domain, &records, policy(mode), &mut work).unwrap();
        assert_eq!(fit.steps.is_empty(), mode == joint::training::Mode::Forward);
        assert!(!fit.target_realizability_certified);
        final_loss.push(fit.final_ce);
        let mut budget = joint::head::PredictionBudget::new(64).unwrap();
        for row in &records {
            assert_eq!(
                fit.head
                    .predict(&row.states, &mut budget, &mut work)
                    .unwrap()
                    .winner,
                row.label
            );
        }
    }
    assert!(final_loss[1] < final_loss[0]);
    assert!(final_loss[2] < final_loss[0]);
}

#[test]
fn boundary_resource_and_training_binding_refusals() {
    assert!(
        serde_json::from_str::<Training>(r#"{"vocabulary":7,"rows":[[0,0,0]],"task":"hidden"}"#)
            .is_err()
    );
    assert!(Evaluator::new(8).is_err());
    assert!(Evaluator::new(1031).is_err());
    let mut t = sample();
    let selection = select(&t, &mut meter()).unwrap();
    t.rows.swap(0, 1);
    assert!(encode(&t, &selection, &mut meter()).is_err());
    t.rows[0][2] = 7;
    assert!(select(&t, &mut meter()).is_err());
    assert!(select(&sample(), &mut Work::new(1).unwrap()).is_err());
}

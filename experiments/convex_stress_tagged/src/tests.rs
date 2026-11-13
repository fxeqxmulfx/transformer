use crate::{Model, Sample, model};

#[test]
fn actual_tag_lift_matches_lean_constant_strips_including_the_intersection() {
    let base = Model::construct(2, vec![0, 1], vec![0, 1], vec![0, 1]).unwrap();
    let axes = vec![None, Some(0), Some(1)];
    let lifted = base.lift(0, &axes, &axes).unwrap();
    assert_eq!((lifted.n, lifted.q), (3, 6));
    assert_eq!(lifted.output_labels, vec![0, 0, 0, 0, 0, 1, 0, 1, 0]);
    assert_eq!(lifted.latent_labels[0], 2);
    let samples: Vec<_> = lifted
        .output_labels
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != 8)
        .map(|(cell, &label)| Sample { cell, label })
        .collect();
    let audit = lifted.audit(&samples).unwrap();
    assert_eq!(audit["training_errors"], 0);
    assert_eq!(audit["raw_frequency_audits"], 6);
    assert_eq!(audit["latent_moment_norms"].as_array().unwrap().len(), 6);
}

#[test]
fn repeated_phases_and_unequal_cardinalities_keep_full_vocabulary_ce() {
    let model = Model::construct(7, vec![2, 2, 4], vec![1, 0, 1], vec![0, 2, 0, 3, 5]).unwrap();
    let mut samples: Vec<_> = model
        .output_labels
        .iter()
        .enumerate()
        .map(|(cell, &label)| Sample { cell, label })
        .collect();
    samples[1].label = (samples[1].label + 1) % 7;
    let audit = model.audit(&samples).unwrap();
    assert_eq!(audit["training_errors"], 1);
    assert_eq!(audit["vocabulary"], 7);
    let ideal = (std::f64::consts::E + 6.).ln() - 1.;
    assert!((audit["train_ce"].as_f64().unwrap() - ideal - 1. / 9.).abs() < 1e-12);
    let axes: Vec<_> = (0..3).map(Some).collect();
    let lifted = model.lift(6, &axes, &axes).unwrap();
    assert_eq!(lifted.output_labels, model.output_labels);
    let raised = lifted.audit(&samples).unwrap();
    assert_eq!(raised["training_errors"], 1);
    assert!(
        (raised["train_ce"].as_f64().unwrap() - audit["train_ce"].as_f64().unwrap()).abs() < 1e-12
    );
}

#[test]
fn raw_coordinate_tampering_and_dimension_bounds_are_checked() {
    assert!(model::memory_bound(257, 771, 1024).unwrap() < 1024 * 1024 * 1024);
    for (n, q, v) in [(258, 3, 2), (3, 772, 2), (3, 3, 1025), (1, 3, 2)] {
        assert!(model::memory_bound(n, q, v).is_err());
    }
    let mut a = Model::construct(3, vec![0, 1], vec![0, 1], vec![0, 1, 2]).unwrap();
    a.latent_labels[0] = 1;
    assert!(a.validate().is_err());
    a.latent_labels[0] = 0;
    a.output_labels[0] = 1;
    assert!(a.validate().is_err());
    a.output_labels[0] = 0;
    a.q = 4;
    assert!(a.validate().is_err());
}

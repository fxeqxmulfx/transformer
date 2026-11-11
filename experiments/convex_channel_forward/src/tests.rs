use crate::model::{Model, Shape, dense_predict};
use convex_channels::{channel::Channel, task};
use convex_recall_compare::data::Sample;

fn sample(values: &[usize], query: usize, label: usize) -> Sample {
    Sample {
        values: values.to_vec(),
        queries: vec![query],
        targets: vec![label],
        tokens: Vec::new(),
        id: 0,
    }
}

#[test]
fn exhaustive_partial_boolean_data_match_brute_force_heads() {
    let shape = Shape {
        queries: 1,
        keys: 2,
        values: 2,
        classes: 2,
    };
    for encoded in 0..81 {
        let mut code = encoded;
        let mut samples = Vec::new();
        for dictionary in 0..4 {
            let label = code % 3;
            code /= 3;
            if label != 0 {
                samples.push(sample(&[dictionary & 1, dictionary >> 1], 0, label - 1));
            }
        }
        // Independent reference enumerates all eight deterministic heads.
        let consistent = (0..2).any(|key| {
            (0..4).any(|outputs| {
                samples
                    .iter()
                    .all(|s| ((outputs >> s.values[key]) & 1) == s.targets[0])
            })
        });
        let fitted = Model::fit(shape, &samples, 0);
        assert_eq!(fitted.is_ok(), consistent, "dataset {encoded}");
        if let Ok((model, visits)) = fitted {
            assert!(visits <= shape.keys * samples.len());
            let dense = model.to_channel();
            assert_eq!(dense.feasibility_error(), 0.0);
            for s in &samples {
                assert_eq!(model.predict(&s.values, &s.queries), s.targets);
            }
            for d in 0..4 {
                let s = sample(&[d & 1, d >> 1], 0, 0);
                assert_eq!(model.logits(&s.values, &s.queries), dense.logits(&s));
            }
        }
    }
}

#[test]
fn lean_boolean_witness_and_held_out_dictionary() {
    let shape = Shape {
        queries: 2,
        keys: 2,
        values: 2,
        classes: 2,
    };
    let samples = [
        sample(&[0, 1], 0, 1),
        sample(&[1, 1], 0, 0),
        sample(&[1, 0], 1, 1),
        sample(&[1, 1], 1, 0),
    ];
    let (model, visits) = Model::fit(shape, &samples, 0).unwrap();
    assert_eq!(model.routes, [0, 1]);
    assert_eq!(model.outputs, [1, 0, 1, 0]);
    assert_eq!(visits, 6);
    assert_eq!(model.predict(&[0, 0], &[1, 0]), [1, 1]);
}

#[test]
fn conflicting_labels_report_failure() {
    let shape = Shape {
        queries: 1,
        keys: 2,
        values: 2,
        classes: 2,
    };
    assert_eq!(
        Model::fit(shape, &[sample(&[0, 0], 0, 0), sample(&[0, 0], 0, 1)], 0),
        Err(0)
    );
}

#[test]
fn uncovered_values_use_only_the_explicit_default() {
    let shape = Shape {
        queries: 1,
        keys: 1,
        values: 3,
        classes: 4,
    };
    let (model, _) = Model::fit(shape, &[sample(&[1], 0, 2)], 3).unwrap();
    assert_eq!(model.outputs, [3, 2, 3]);
}

#[test]
fn compaction_is_exact_and_rejects_nonvertices() {
    let model = Model {
        shape: Shape {
            queries: 1,
            keys: 2,
            values: 2,
            classes: 2,
        },
        routes: vec![1],
        outputs: vec![1, 0],
    };
    let mut dense = model.to_channel();
    assert_eq!(Model::compact(&dense), Some(model));
    let at = dense.index(0, 1, 0, 1);
    dense.weights[at] = 1.0 - 1e-12;
    assert!(Model::compact(&dense).is_none());
    assert!(Model::compact(&Channel::new(1, 2, 2, 2)).is_none());
    dense.weights.fill(0.0);
    let first = dense.index(0, 0, 0, 0);
    let second = dense.index(0, 1, 1, 1);
    dense.weights[first] = 1.0;
    dense.weights[second] = 1.0;
    assert!(Model::compact(&dense).is_none());
}

#[test]
fn independent_seed_generates_absent_values_and_matches_dense_forward() {
    let task = task::make(73, true);
    let shape = Shape {
        queries: task::KEYS,
        keys: task::KEYS,
        values: task::VALUES,
        classes: task::CLASSES,
    };
    let (model, _) = Model::fit(shape, &task.data.train, 0).unwrap();
    let dense = model.to_channel();
    for mut s in task::all_dictionaries(&task) {
        let expected = s.targets.clone();
        s.targets.fill(usize::MAX);
        assert_eq!(model.predict(&s.values, &s.queries), expected);
        assert_eq!(dense_predict(&dense, &s), expected);
        assert!(expected.iter().all(|c| !s.tokens.contains(c)));
    }
}

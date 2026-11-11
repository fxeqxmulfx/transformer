use crate::streaming::Streaming;
use convex_channel_forward::model::{Model, Shape};
use convex_channels::task;
use convex_recall_compare::data::Sample;

fn shape() -> Shape {
    Shape {
        queries: 1,
        keys: 2,
        values: 2,
        classes: 2,
    }
}

fn sample(values: [usize; 2], label: usize) -> Sample {
    Sample {
        tokens: vec![],
        targets: vec![label],
        values: values.to_vec(),
        queries: vec![0],
        id: 0,
    }
}

fn reference(s: Shape, samples: &[Sample], default: usize) -> Result<Model, usize> {
    Model::fit(s, samples, default).map(|(m, _)| m)
}

#[test]
fn all_partial_boolean_data_and_partitions() {
    // Each of the four dictionaries is absent, labeled 0, or labeled 1.
    // Enumerate all 8 deterministic heads independently to check realizability.
    for code in 0..81usize {
        let mut remaining = code;
        let mut samples = Vec::new();
        for dictionary in 0..4 {
            let choice = remaining % 3;
            remaining /= 3;
            if choice > 0 {
                samples.push(sample([dictionary / 2, dictionary % 2], choice - 1));
            }
        }
        let feasible: Vec<_> = (0..2)
            .flat_map(|k| (0..4).map(move |table| (k, [table & 1, table >> 1])))
            .filter(|(k, table)| samples.iter().all(|s| table[s.values[*k]] == s.targets[0]))
            .collect();
        for default in 0..2 {
            let expected = reference(shape(), &samples, default);
            assert_eq!(expected.is_ok(), !feasible.is_empty());
            if let Ok(model) = &expected {
                assert_eq!(model.routes[0], feasible[0].0);
                assert!(feasible.iter().any(|(k, table)| {
                    *k == model.routes[0] && table.as_slice() == model.outputs
                }));
            }
            for cuts in 0..(1 << samples.len().saturating_sub(1)) {
                let mut state = Streaming::new(shape(), default);
                let mut start = 0;
                for end in 1..=samples.len() {
                    if end == samples.len() || (cuts & (1 << (end - 1))) != 0 {
                        state.update_batch(&[]);
                        state.update_batch(&samples[start..end]);
                        let saved = state.clone();
                        assert_eq!(
                            state.snapshot(),
                            reference(shape(), &samples[..end], default)
                        );
                        assert_eq!(state, saved, "snapshot changed records");
                        start = end;
                    }
                }
                state.update_batch(&[]);
                assert_eq!(state.snapshot(), expected);
                assert_eq!(state.candidate_visits(), 2 * samples.len());
            }
        }
    }
}

#[test]
fn late_conflict_preserves_alternative_history() {
    // Lean's prefix_selects_first and later_batch_changes_key.
    let mut state = Streaming::new(shape(), 0);
    state.update_batch(&[sample([0, 0], 1)]);
    assert_eq!(state.snapshot().unwrap().routes, vec![0]);
    state.update_batch(&[]);
    state.update_batch(&[sample([0, 1], 0)]);
    let model = state.snapshot().unwrap();
    assert_eq!(model.routes, vec![1]);
    assert_eq!(model.outputs, vec![1, 0]);
    assert_eq!(model.predict(&[1, 0], &[0]), vec![1]);
}

#[test]
fn snapshot_defaults_are_not_training_labels() {
    let s = Shape { keys: 1, ..shape() };
    let mut first = sample([0, 0], 1);
    first.values = vec![0];
    let mut state = Streaming::new(s, 0);
    state.update_batch(&[first.clone()]);
    assert_eq!(state.snapshot().unwrap().outputs, vec![1, 0]);
    first.values[0] = 1;
    state.update_batch(&[first]);
    assert_eq!(state.snapshot().unwrap().outputs, vec![1, 1]);
}

#[test]
fn empty_batches_and_failed_candidates_are_persistent() {
    let mut state = Streaming::new(shape(), 1);
    let before = state.clone();
    state.update_batch(&[]);
    assert_eq!(state, before);
    assert_eq!(state.snapshot().unwrap().outputs, vec![1, 1]);
    state.update_batch(&[sample([0, 0], 0)]);
    state.update_batch(&[sample([0, 0], 1)]);
    assert_eq!(state.snapshot(), Err(0));
    state.update_batch(&[sample([1, 1], 0)]);
    assert_eq!(state.snapshot(), Err(0));
    assert_eq!(state.candidate_visits(), 6);
    assert_eq!(state.state_payload_bytes(), before.state_payload_bytes());
}

#[test]
fn mixed_query_order_duplicates_and_unobserved_rows() {
    let s = Shape {
        queries: 3,
        ..shape()
    };
    let mut a = sample([0, 0], 1);
    a.queries = vec![2, 0, 2];
    a.targets = vec![1, 0, 1];
    let mut b = sample([0, 1], 0);
    b.queries = vec![0, 2];
    b.targets = vec![0, 0];
    let data = [a, b];
    let mut state = Streaming::new(s, 1);
    for (i, row) in data.iter().enumerate() {
        state.update_batch(std::slice::from_ref(row));
        assert_eq!(state.snapshot(), reference(s, &data[..=i], 1));
    }
    let model = state.snapshot().unwrap();
    assert_eq!(model.routes, vec![0, 0, 1]);
    assert_eq!(&model.outputs[2..4], &[1, 1]);
    assert_eq!(state.candidate_visits(), 10);
}

#[test]
fn mqar_partial_batches_match_each_prefix() {
    let data = task::make(0, true);
    let s = Shape {
        queries: task::KEYS,
        keys: task::KEYS,
        values: task::VALUES,
        classes: task::CLASSES,
    };
    let mut state = Streaming::new(s, 0);
    let mut count = 0;
    for batch in data.data.train.chunks(7) {
        state.update_batch(batch);
        count += batch.len();
        assert_eq!(state.snapshot(), reference(s, &data.data.train[..count], 0));
    }
    let model = state.snapshot().unwrap();
    for sample in task::all_dictionaries(&data) {
        assert_eq!(
            model.predict(&sample.values, &sample.queries),
            sample.targets
        );
    }
    assert_eq!(state.candidate_visits(), 4096);
    assert_eq!(
        state.state_payload_bytes(),
        128 * size_of::<Option<usize>>() + 16
    );
}

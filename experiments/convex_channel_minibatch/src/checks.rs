//! Exact model/prefix checks and evaluation; no task rule enters the learner.

use crate::bench;
use convex_channel_forward::model::Model;
use convex_channel_minibatch::streaming::Streaming;
use convex_channels::task::{self, Task};
use convex_recall_compare::data::Sample;
use serde_json::{Value, json};

pub fn evaluate(model: &Model, samples: &[Sample]) -> Value {
    let (mut correct, mut answers, mut dictionaries) = (0, 0, 0);
    for sample in samples {
        let predictions = model.predict(&sample.values, &sample.queries);
        assert_eq!(predictions.len(), sample.targets.len());
        correct += predictions
            .iter()
            .zip(&sample.targets)
            .filter(|(p, t)| p == t)
            .count();
        answers += sample.targets.len();
        dictionaries += usize::from(predictions == sample.targets);
    }
    json!({"correct":correct,"answers":answers,"accuracy":correct as f64/answers as f64,
        "correct_dictionaries":dictionaries,"dictionaries":samples.len()})
}

pub fn model_json(model: &Model) -> Value {
    json!({"routes":model.routes,"outputs":model.outputs,"payload_bytes":model.payload_bytes()})
}

pub fn read_model(saved: &Value) -> Model {
    let s = bench::shape();
    let integers = |key: &str| {
        saved[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| usize::try_from(v.as_u64().unwrap()).unwrap())
            .collect::<Vec<_>>()
    };
    let model = Model {
        shape: s,
        routes: integers("routes"),
        outputs: integers("outputs"),
    };
    assert_eq!(model.routes.len(), s.queries);
    assert_eq!(model.outputs.len(), s.queries * s.values);
    assert!(model.routes.iter().all(|&k| k < s.keys));
    assert!(model.outputs.iter().all(|&c| c < s.classes));
    assert_eq!(saved["payload_bytes"], model.payload_bytes());
    model
}

pub fn inspect(task: &Task, batch_size: usize, model: &Model, reference: &Model) -> Value {
    assert_eq!(model, reference);
    assert_eq!(model.to_channel().weights, reference.to_channel().weights);
    let mut state = Streaming::new(bench::shape(), 0);
    let (mut prefix, mut snapshots) = (0, 0);
    for batch in task.data.train.chunks(batch_size) {
        state.update_batch(batch);
        prefix += batch.len();
        let expected = Model::fit(bench::shape(), &task.data.train[..prefix], 0).map(|(m, _)| m);
        assert_eq!(state.snapshot(), expected);
        snapshots += 1;
    }
    assert_eq!(state.snapshot().unwrap(), *model);
    let reversed: Vec<_> = task.data.train.iter().rev().cloned().collect();
    assert_eq!(bench::fit(&reversed, batch_size).0, *model);
    let visits = task
        .data
        .train
        .iter()
        .map(|s| s.targets.len())
        .sum::<usize>()
        * task::KEYS;
    assert_eq!(state.candidate_visits(), visits);
    json!({"prefix_snapshots":snapshots,"exact_model":true,"exact_tensor":true,
        "reversed_order_agrees":true,"candidate_visits":visits,
        "state_payload_bytes":state.state_payload_bytes(),
        "train":evaluate(model,&task.data.train),"test":evaluate(model,&task.data.test),
        "all_dictionaries":evaluate(model,&task::all_dictionaries(task))})
}

/// Ported from convex_channel_forward at 3e0e2cb1. Accommodates JSON f64
/// round trips; categorical models and the reconstructed tensor are checked exactly.
pub fn equivalent(saved: &Value, actual: &Value) {
    match (saved, actual) {
        (Value::Object(a), Value::Object(b)) => {
            assert_eq!(a.len(), b.len());
            for (key, value) in b {
                equivalent(a.get(key).unwrap(), value);
            }
        }
        (Value::Array(a), Value::Array(b)) => {
            assert_eq!(a.len(), b.len());
            for (x, y) in a.iter().zip(b) {
                equivalent(x, y);
            }
        }
        (Value::Number(a), Value::Number(b)) => {
            let (x, y) = (a.as_f64().unwrap(), b.as_f64().unwrap());
            assert!((x - y).abs() <= 1e-12 * x.abs().max(y.abs()).max(1.0));
        }
        _ => assert_eq!(saved, actual),
    }
}

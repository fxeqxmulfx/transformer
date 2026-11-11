//! Independent evaluation and replay of the actual saved predictions.

use convex_channel_forward::model::{Model, argmax};
use convex_channels::{
    channel::Channel,
    task::{self, Task},
};
use convex_recall_compare::{convex::ce_gradient, data::Sample};
use serde_json::{Value, json};

pub fn evaluate(channel: &Channel, samples: &[Sample]) -> Value {
    let (mut correct, mut count, mut all_correct) = (0usize, 0usize, 0usize);
    let mut loss = 0.0;
    for sample in samples {
        let z = channel.logits(sample);
        let mut all = true;
        for (row, &target) in z.chunks_exact(channel.classes).zip(&sample.targets) {
            let right = argmax(row) == target;
            correct += usize::from(right);
            count += 1;
            all &= right;
            loss += ce_gradient(row, target).0;
        }
        all_correct += usize::from(all);
    }
    json!({"accuracy":correct as f64/count as f64,"ce":loss/count as f64,
        "all_queries_correct":all_correct as f64/samples.len() as f64,"queries":count})
}

pub fn inspect(channel: &Channel, task: &Task) -> Value {
    let full = task::all_dictionaries(task);
    let references: Vec<_> = task.data.train.iter().collect();
    json!({"train":evaluate(channel,&task.data.train),"test":evaluate(channel,&task.data.test),
        "all_dictionaries":evaluate(channel,&full),"feasibility_error":channel.feasibility_error(),
        "fw_gap":channel.gap(&references),"rule_error":task::rule_error(channel,task),
        "spectrum":task::spectrum(task,Some(channel)),
        "tensor_payload_bytes":channel.weights.len()*size_of::<f64>()})
}

pub fn check_compact(model: &Model, channel: &Channel, task: &Task) {
    assert_eq!(model.to_channel().weights, channel.weights);
    for sample in task::all_dictionaries(task) {
        let z = channel.logits(&sample);
        assert_eq!(model.logits(&sample.values, &sample.queries), z);
        let expected: Vec<_> = z.chunks_exact(channel.classes).map(argmax).collect();
        assert_eq!(model.predict(&sample.values, &sample.queries), expected);
    }
}

/// Checks CF4 independently from the learner; task rules are evaluation-only.
pub fn identifying_data(task: &Task) -> bool {
    (0..task::KEYS).all(|q| {
        let observations: Vec<_> = task
            .data
            .train
            .iter()
            .flat_map(|s| {
                s.queries
                    .iter()
                    .zip(&s.targets)
                    .filter_map(move |(&query, &label)| (q == query).then_some((&s.values, label)))
            })
            .collect();
        let covered = (0..task::VALUES).all(|v| observations.iter().any(|&(d, _)| d[q] == v));
        let labels = observations
            .iter()
            .all(|&(d, c)| task.rule[q * task::VALUES + d[q]] == c);
        let separated = (0..task::KEYS).all(|k| {
            k == q
                || observations
                    .iter()
                    .any(|&(a, ca)| observations.iter().any(|&(b, cb)| a[k] == b[k] && ca != cb))
        });
        covered && labels && separated
    })
}

pub fn model_json(model: &Model) -> Value {
    json!({"routes":model.routes,"outputs":model.outputs,"payload_bytes":model.payload_bytes()})
}

pub fn read_model(value: &Value) -> Model {
    let shape = crate::bench::shape();
    let integers = |name: &str| {
        value[name]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect::<Vec<_>>()
    };
    let model = Model {
        shape,
        routes: integers("routes"),
        outputs: integers("outputs"),
    };
    assert_eq!(model.routes.len(), shape.queries);
    assert_eq!(model.outputs.len(), shape.queries * shape.values);
    assert!(model.routes.iter().all(|&k| k < shape.keys));
    assert!(model.outputs.iter().all(|&c| c < shape.classes));
    assert_eq!(
        value["payload_bytes"].as_u64().unwrap() as usize,
        model.payload_bytes()
    );
    model
}

pub fn read_channel(value: &Value) -> Channel {
    let s = crate::bench::shape();
    let mut channel = Channel::new(s.queries, s.keys, s.values, s.classes);
    let weights: Vec<_> = value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    assert_eq!(weights.len(), channel.weights.len());
    assert!(weights.iter().all(|v| v.is_finite()));
    channel.weights = weights;
    channel
}

/// JSON floating-point round trips are checked to a fixed 1e-12 relative tolerance.
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
            assert!(
                (x - y).abs() <= 1e-12 * x.abs().max(y.abs()).max(1.0),
                "{x} != {y}"
            );
        }
        _ => assert_eq!(saved, actual),
    }
}

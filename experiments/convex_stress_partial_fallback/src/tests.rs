use super::*;
use causal::metrics::Predictor;
use serde_json::{Value, json};
mod boundaries;
mod gradients;
mod modes;
mod saved;

fn request() -> causal::Request {
    causal::Request {
        input: Input {
            vocabulary: 7,
            context: 5,
            observations: vec![
                causal::Observation {
                    tokens: vec![0, 0, 6],
                    label: 2,
                },
                causal::Observation {
                    tokens: vec![0, 1, 6],
                    label: 3,
                },
                causal::Observation {
                    tokens: vec![1, 0, 6],
                    label: 4,
                },
                causal::Observation {
                    tokens: vec![1, 1, 6],
                    label: 5,
                },
            ],
        },
        policy: causal::Policy::default(),
    }
}
fn candidate(x: &causal::Request) -> Result<causal::model::Model> {
    Ok(causal::model::initialize(&causal::discovery::run(x.clone())?)?.model)
}
fn count_value(v: usize, context: usize) -> Value {
    let rows = |weight: u64, shift: usize| {
        (0..v + 2 + context)
            .map(|i| {
                let class = 1 + (i + shift) % (v - 1);
                let mut row = serde_json::Map::new();
                row.insert(class.to_string(), json!(weight));
                Value::Object(row)
            })
            .collect::<Vec<_>>()
    };
    json!({"features":{"base":{"vocab":v,"context":context}},"scale":7.3,
        "shared":{"keys":{"2":{"weight":3,"counts":rows(3,0)}},"active":[2],"total":3},
        "local":{"1":{"keys":{"2":{"weight":2,"counts":rows(2,1)}},"active":[2],"total":2}},"total":7})
}
fn parent(v: usize, context: usize) -> Result<Parent> {
    Parent::from_saved("counts", count_value(v, context))
}
fn batch_loss(model: &Model, input: &Input, ids: &[usize]) -> Result<f64> {
    let mut total = 0.;
    for &id in ids {
        let s = &input.observations[id];
        let p: Vec<_> = s.tokens.iter().map(|&x| x as i64).collect();
        total += causal::training::ce(&model.logits(&p)?, s.label);
    }
    Ok(total / ids.len() as f64)
}

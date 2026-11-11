//! Post-run representation witness, not a trained model or benchmark arm.
//! Source: the frozen numeric generator, using lags 64 and 32. Its hand-written
//! parameters test whether the existing additive features can classify all outputs.
use convex_stress_minibatch::{
    Result,
    channel::{Channel, QueryChannel},
    data,
    features::Adapter,
    metrics,
};
use serde_json::{Value, json};
use std::collections::BTreeMap;
struct Scaled<'a> {
    model: &'a Channel,
    scale: f64,
}
impl metrics::Predictor for Scaled<'_> {
    fn vocab(&self) -> usize {
        self.model.adapter.vocab
    }
    fn logits(&self, prefix: &[i64]) -> Vec<f64> {
        self.model
            .logits(prefix)
            .into_iter()
            .map(|z| z * self.scale)
            .collect()
    }
}
fn main() -> Result<()> {
    let suite: Value = serde_json::from_str(include_str!("../../../convex_stress/suite.json"))?;
    let row = suite["tasks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["name"] == "gpt-probe-branch-amplitudes")
        .unwrap();
    let task = data::build(row)?;
    let adapter = Adapter {
        vocab: task.vocab,
        context: task
            .splits
            .values()
            .flatten()
            .map(|e| e.tokens.len())
            .max()
            .unwrap(),
    };
    let classes = vec![0, 17, 47, 49, 55, 57];
    let mut rows = BTreeMap::new();
    for (k, v, outputs) in [
        (64, 48, vec![(47, 0.25), (49, 0.25)]),
        (64, 56, vec![(55, 0.25), (57, 0.25)]),
        (32, 48, vec![(47, 0.25), (55, 0.25)]),
        (32, 56, vec![(49, 0.25), (57, 0.25)]),
        (32, 18, vec![(17, 0.5)]),
    ] {
        let mut weights = vec![0.; classes.len()];
        for (c, w) in outputs {
            weights[classes.binary_search(&c).unwrap()] = w;
        }
        rows.insert(k * adapter.values() + v, weights);
    }
    let block = QueryChannel {
        classes,
        routes: [(32, 0.5), (64, 0.5)].into(),
        rows,
    };
    let witness = Channel {
        adapter,
        queries: (0..adapter.vocab).map(|q| (q, block.clone())).collect(),
    };
    let trained: Channel = serde_json::from_slice(&std::fs::read(
        std::env::args()
            .nth(1)
            .ok_or("supply saved v1 branch channel")?,
    )?)?;
    let mut observations = vec![];
    for (name, model) in [
        ("representation_witness", &witness),
        ("trained_v1", &trained),
    ] {
        for scale in [1., (adapter.vocab as f64).ln(), 10.] {
            let predictor = Scaled { model, scale };
            let mut splits = serde_json::Map::new();
            for (split, examples) in &task.splits {
                let scores = metrics::evaluate(&predictor, examples)?;
                splits.insert(split.clone(),json!({"ce":scores["loss"],"exact":scores["sequence_accuracy"],"token":scores["token_accuracy"]}));
            }
            observations.push(json!({"model":name,"fixed_logit_scale":scale,"splits":splits}));
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"witness_feasibility_error":witness.feasibility_error()?,"observations":observations})
        )?
    );
    Ok(())
}
